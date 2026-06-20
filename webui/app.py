import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.utils
import traceback

# 将项目根目录加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入真实模型
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    REAL_MODEL_AVAILABLE = True
except ImportError:
    REAL_MODEL_AVAILABLE = False
    print("⚠️ 真实模型不可用，将使用模拟模式进行演示。")

# 模型配置（与前端下拉菜单一致）
MODEL_OPTIONS = [
    {
        "id": "small",
        "label": "Kronos-small (推荐·24.7M·上下文512)",
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    {
        "id": "base",
        "label": "Kronos-base (高精度·102.3M·上下文512·需≥8GB显存)",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    {
        "id": "mini",
        "label": "Kronos-mini (最快·4.1M·上下文2048·可跑CPU)",
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
]

app = Flask(__name__, template_folder='templates')
CORS(app)

# 全局变量
predictor = None
tokenizer = None
model = None
current_model_id = None
current_model_label = "未加载"
DATA_DIR = os.path.join(project_root, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 模拟预测器（当真实模型不可用时使用）
class MockPredictor:
    def __init__(self, model_id="small"):
        self.model_id = model_id
        self.max_context = 512
        self.clip = 5
        self.device = "cpu"
        print(f"🟡 使用模拟预测器 (模型: {model_id})")

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False):
        # 生成随机预测数据
        np.random.seed(42)
        cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        pred_data = np.random.randn(pred_len, 6).cumsum(axis=0) + df[['open','high','low','close','volume','amount']].values[-1]
        pred_df = pd.DataFrame(pred_data, columns=cols, index=y_timestamp)
        return pred_df

def load_data_file(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        df.columns = [c.lower().strip() for c in df.columns]
        required = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required):
            return None, f"Missing required columns: {required}"
        tc = None
        for col in ['timestamps', 'timestamp', 'date', 'time', 'datetime']:
            if col in df.columns:
                tc = col
                break
        if tc is None:
            df['timestamps'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        else:
            df['timestamps'] = pd.to_datetime(df[tc])
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None,
        'current_model': current_model_id,
        'model_label': current_model_label,
        'real_model': REAL_MODEL_AVAILABLE
    })

@app.route('/api/models')
def list_models():
    return jsonify({'models': MODEL_OPTIONS})

@app.route('/api/data-files')
def get_data_files():
    files = []
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.endswith(('.csv', '.feather')):
                fp = os.path.join(DATA_DIR, f)
                sz = os.path.getsize(fp)
                files.append({'name': f, 'path': fp, 'size': f"{sz/1024:.1f}KB" if sz<1024 * 1024 else f"{sz/(1024 * 1024):.1f}MB"})
    return jsonify(files)

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not (f.filename.endswith('.csv') or f.filename.endswith('.feather')):
        return jsonify({'error': 'Only .csv/.feather'}), 400
    from werkzeug.utils import secure_filename
    fn = secure_filename(f.filename)
    sp = os.path.join(DATA_DIR, fn)
    f.save(sp)
    df, err = load_data_file(sp)
    if err:
        os.remove(sp)
        return jsonify({'error': err}), 400
    return jsonify({'success': True, 'filepath': sp, 'filename': fn, 'rows': len(df)})

@app.route('/api/load-model', methods=['POST'])
def load_model():
    global predictor, tokenizer, model, current_model_id, current_model_label
    try:
        body = request.get_json(force=True, silent=True) or {}
        model_id = body.get('model_id', 'small')
        cfg = next((m for m in MODEL_OPTIONS if m['id'] == model_id), MODEL_OPTIONS[0])

        if not REAL_MODEL_AVAILABLE:
            # 使用模拟预测器
            predictor = MockPredictor(model_id)
            current_model_id = cfg['id']
            current_model_label = cfg['label']
            return jsonify({
                'success': True,
                'model': cfg['id'],
                'label': cfg['label'],
                'max_context': cfg['max_context'],
                'device': 'cpu',
                'message': f'模拟模式 - {cfg["label"]}'
            })

        # 真实模型加载
        from model import Kronos, KronosTokenizer, KronosPredictor
        tok = KronosTokenizer.from_pretrained(cfg['tokenizer_id'])
        mdl = Kronos.from_pretrained(cfg['model_id'])
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        predictor = KronosPredictor(mdl, tok, device=device, max_context=cfg['max_context'])
        tokenizer = tok
        model = mdl
        current_model_id = cfg['id']
        current_model_label = cfg['label']

        return jsonify({
            'success': True,
            'model': cfg['id'],
            'label': cfg['label'],
            'max_context': cfg['max_context'],
            'device': device,
            'message': f'Loaded {cfg["label"]} on {device}'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded. Load a model first.'}), 400

        data = request.get_json()
        file_path = data.get('filepath') or data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        mode = data.get('mode', 'forecast')
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))

        mc = predictor.max_context or 512
        lookback = min(lookback, mc)

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400

        df, err = load_data_file(file_path)
        if err:
            return jsonify({'error': err}), 400
        if len(df) < lookback + (pred_len if mode == 'backtest' else 0):
            return jsonify({'error': f'Data too short (need ≥{lookback+pred_len}, got {len(df)})'}), 400

        # 固定随机种子以保证可复现性
        torch.manual_seed(42)
        np.random.seed(42)

        # 准备输入数据
        x_df = df.iloc[-lookback:].copy()
        for col in ['open','high','low','close']:
            x_df[col] = pd.to_numeric(x_df[col], errors='coerce')
        if 'volume' not in x_df.columns: x_df['volume'] = 0.0
        if 'amount' not in x_df.columns: x_df['amount'] = x_df['volume'] * x_df['close']

        x_ts = x_df['timestamps'].reset_index(drop=True)
        last = x_ts.iloc[-1]
        td = x_ts.diff().median()
        if pd.isna(td) or (hasattr(td,'total_seconds') and td.total_seconds()==0) or (td==pd.Timedelta(0)):
            td = pd.Timedelta(days=1)
        y_ts = pd.date_range(start=last + td, periods=pred_len, freq=td)

        # 调用预测
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=pred_len,
            T=temperature, top_p=top_p, sample_count=sample_count, verbose=False
        )

        # 生成图表
        def _chart(hist_part, p_df, title):
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist_part['timestamps'], open=hist_part['open'], high=hist_part['high'],
                low=hist_part['low'], close=hist_part['close'],
                name='Historical', increasing_line_color='#1a9850', decreasing_line_color='#d73027', line=dict(width=1.5)))
            fig.add_trace(go.Candlestick(
                x=p_df.index, open=p_df['open'], high=p_df['high'], low=p_df['low'], close=p_df['close'],
                name='Prediction', increasing_line_color='#00cc96', decreasing_line_color='#ff7f0e', line=dict(width=2)))
            fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', template='plotly_white', height=500)
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        if mode == 'backtest':
            actual_df = df.iloc[lookback:lookback+pred_len].copy()
            chart_json = _chart(df.iloc[-lookback:], pred_df, f'Kronos Backtest - {lookback}h + {pred_len}p vs Actual')
        else:
            chart_json = _chart(df.iloc[-lookback:], pred_df, f'Kronos Forecast - {lookback}h + {pred_len}p')

        results = []
        for ts, r in pred_df.iterrows():
            results.append({'timestamp': pd.Timestamp(ts).isoformat(),
                            'open': float(r['open']), 'high': float(r['high']),
                            'low': float(r['low']), 'close': float(r['close']),
                            'volume': float(r.get('volume',0)), 'amount': float(r.get('amount',0))})

        has_cmp = (mode == 'backtest')
        metrics = {'mae':0,'rmse':0,'mape':0}
        if has_cmp:
            ac = df.iloc[lookback:lookback+pred_len]['close'].values
            pc = pred_df['close'].values[:len(ac)]
            e = np.abs(pc-ac)
            metrics['mae'] = round(float(np.mean(e)),4)
            metrics['rmse'] = round(float(np.sqrt(np.mean(e**2))),4)
            metrics['mape'] = round(float(np.mean(e/(np.abs(ac)+1e-10))*100),2)

        return jsonify({
            'success': True,
            'chart': chart_json,
            'prediction_results': results,
            'has_comparison': has_cmp,
            'metrics': metrics,
            'model_used': current_model_label,  # 返回使用的模型名称
            'message': f'{current_model_id} predicted {pred_len} pts'
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Kronos WebUI 启动")
    print(f"   真实模型可用: {REAL_MODEL_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=7070)