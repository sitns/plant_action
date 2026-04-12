import argparse
import asyncio
import json
import threading
import time
import webbrowser
from collections import deque
from typing import Any

from flask import Flask, jsonify, render_template_string, request

try:
    from bleak import BleakClient, BleakScanner
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: bleak. Install with: pip install bleak flask"
    ) from exc

DEVICE_NAME = "ESP32-EnvSensor"
SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
CHAR_UUID = "abcdefab-1234-5678-9abc-def012345678"
FAN_CHAR_UUID = "fedcba98-7654-3210-fedc-ba9876543210"
MAX_POINTS = 240

app = Flask(__name__)
state_lock = threading.Lock()
history = deque(maxlen=MAX_POINTS)
ble_loop: asyncio.AbstractEventLoop | None = None
ble_client: BleakClient | None = None
desired_fan_percent = 0
ble_status = {
    "state": "idle",
    "message": "not started",
    "last_update": 0.0,
}


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Environmental Sensor Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #f2efe8;
      --panel: #fffaf3;
      --ink: #1f2a24;
      --grid: #d6d1c4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, #fff8ea 0%, transparent 35%),
        radial-gradient(circle at 95% 100%, #e3f4e9 0%, transparent 45%),
        var(--bg);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }
    .wrap {
      width: min(1100px, 100%);
      background: var(--panel);
      border: 1px solid #e8e1d5;
      border-radius: 16px;
      box-shadow: 0 14px 40px rgba(48, 45, 36, 0.12);
      overflow: hidden;
    }
    .head {
      padding: 18px 20px;
      background: linear-gradient(120deg, #ecf8f1, #fff8eb 60%);
      border-bottom: 1px solid #e8e1d5;
    }
    .head h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0.4px;
    }
    .status {
      margin-top: 8px;
      font-size: 14px;
      color: #47554d;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      padding: 14px 20px 12px;
      align-items: center;
      border-bottom: 1px solid #ece6db;
      background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(255,250,243,0.9));
    }
    .control-group {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .control-label {
      font-size: 12px;
      color: #6f7a73;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border: 1px solid #d9d1c4;
      border-radius: 999px;
      background: #fff;
      font-size: 13px;
      color: #33413a;
    }
    .chip input { margin: 0; }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      padding: 16px 20px 8px;
    }
    .card {
      background: #fff;
      border: 1px solid #ece6db;
      border-radius: 12px;
      padding: 12px;
      min-width: 0;
    }
    .card-wide {
      grid-column: span 2;
    }
    .k {
      font-size: 12px;
      color: #6f7a73;
      margin-bottom: 6px;
    }
    .v {
      font-size: 28px;
      font-weight: 700;
      word-break: break-word;
    }
    .status-text {
      font-size: 17px;
      line-height: 1.4;
    }
    .time-text {
      font-size: 18px;
    }
    .fan-card {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .fan-value-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .fan-slider {
      width: 100%;
      accent-color: #c25b2f;
    }
    .fan-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .fan-button {
      border: 0;
      border-radius: 999px;
      padding: 8px 14px;
      font: inherit;
      color: #fffaf3;
      background: #2f6f5f;
      cursor: pointer;
    }
    .fan-button.secondary {
      background: #b16b2d;
    }
    .fan-hint {
      font-size: 12px;
      color: #6f7a73;
    }
    .chart {
      padding: 14px 20px 20px;
    }
    .chart-stack {
      display: grid;
      gap: 14px;
    }
    .chart-card {
      background: #fff;
      border: 1px solid #ece6db;
      border-radius: 12px;
      overflow: hidden;
    }
    .chart-title {
      padding: 10px 12px;
      border-bottom: 1px solid #ece6db;
      font-size: 13px;
      color: #55625a;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      background: #fffaf5;
    }
    canvas {
      width: 100% !important;
      height: 380px !important;
      background: #fff;
      border: 1px solid #ece6db;
      border-radius: 12px;
    }
    .chart-card canvas {
      border: 0;
      border-radius: 0;
      height: 260px !important;
    }
    .hidden {
      display: none !important;
    }
    @media (max-width: 860px) {
      .grid {
        grid-template-columns: 1fr 1fr;
      }
      .card-wide {
        grid-column: span 2;
      }
    }
    @media (max-width: 760px) {
      .controls { align-items: flex-start; }
      .grid { grid-template-columns: 1fr; }
      .card-wide { grid-column: span 1; }
      .v { font-size: 24px; }
      canvas { height: 300px !important; }
      .chart-card canvas { height: 220px !important; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1>Environmental Live Dashboard</h1>
      <div class="status" id="status">BLE status: loading...</div>
    </div>

    <div class="controls">
      <div class="control-group">
        <div class="control-label">Chart Layout</div>
        <label class="chip"><input type="radio" name="chartMode" value="single" checked /> Single Chart</label>
        <label class="chip"><input type="radio" name="chartMode" value="split" /> Split Charts</label>
      </div>
      <div class="control-group">
        <div class="control-label">Visible Metrics</div>
        <label class="chip"><input type="checkbox" data-metric="soil" checked /> Soil</label>
        <label class="chip"><input type="checkbox" data-metric="temperature" checked /> Temperature</label>
        <label class="chip"><input type="checkbox" data-metric="humidity" checked /> Humidity</label>
        <label class="chip"><input type="checkbox" data-metric="wind" checked /> Wind</label>
        <label class="chip"><input type="checkbox" data-metric="light" checked /> Light</label>
        <label class="chip"><input type="checkbox" data-metric="pressure" checked /> Pressure</label>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="k">Air Temperature</div>
        <div class="v" id="airTemperature">-- C</div>
      </div>
      <div class="card">
        <div class="k">Air Humidity</div>
        <div class="v" id="airHumidity">-- %</div>
      </div>
      <div class="card">
        <div class="k">Soil Analog</div>
        <div class="v" id="soilHumidity">--</div>
      </div>
      <div class="card">
        <div class="k">Wind Speed</div>
        <div class="v" id="windSpeed">-- m/s</div>
      </div>
      <div class="card">
        <div class="k">Light Intensity</div>
        <div class="v" id="lightLux">-- lx</div>
      </div>
      <div class="card">
        <div class="k">Pressure</div>
        <div class="v" id="pressureHpa">-- hPa</div>
      </div>
      <div class="card">
        <div class="k">Fan PWM</div>
        <div class="v" id="fanPwmValue">0 %</div>
      </div>
      <div class="card card-wide fan-card">
        <div class="k">Fan Control</div>
        <div class="fan-value-row">
          <div class="v time-text" id="fanSliderValue">0 %</div>
          <div class="fan-hint" id="fanControlHint">Use BLE to set fan speed</div>
        </div>
        <input class="fan-slider" id="fanPwmSlider" type="range" min="0" max="100" step="1" value="0" />
        <div class="fan-actions">
          <button class="fan-button" id="applyFanPwm" type="button">Apply</button>
          <button class="fan-button secondary" id="stopFanPwm" type="button">Stop Fan</button>
        </div>
      </div>
      <div class="card card-wide">
        <div class="k">Sensor Status</div>
        <div class="v status-text" id="sensorState">--</div>
      </div>
      <div class="card">
        <div class="k">Last Packet Time</div>
        <div class="v time-text" id="packetTime">--</div>
      </div>
    </div>

    <div class="chart">
      <div id="singleChartWrap">
        <canvas id="envChart"></canvas>
      </div>
      <div id="splitChartWrap" class="chart-stack hidden">
        <div class="chart-card">
          <div class="chart-title">Soil Analog</div>
          <canvas id="soilChart"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">Temperature And Humidity</div>
          <canvas id="climateChart"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">Light Intensity</div>
          <canvas id="lightChart"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">Wind Speed</div>
          <canvas id="windChart"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">Pressure</div>
          <canvas id="pressureChart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <script>
    const metricToggles = Array.from(document.querySelectorAll('[data-metric]'));
    const chartModeInputs = Array.from(document.querySelectorAll('input[name="chartMode"]'));
    const airTemperatureEl = document.getElementById('airTemperature');
    const airHumidityEl = document.getElementById('airHumidity');
    const soilHumidityEl = document.getElementById('soilHumidity');
    const windSpeedEl = document.getElementById('windSpeed');
    const lightLuxEl = document.getElementById('lightLux');
    const pressureHpaEl = document.getElementById('pressureHpa');
    const fanPwmValueEl = document.getElementById('fanPwmValue');
    const sensorStateEl = document.getElementById('sensorState');
    const packetTimeEl = document.getElementById('packetTime');
    const statusEl = document.getElementById('status');
    const singleChartWrap = document.getElementById('singleChartWrap');
    const splitChartWrap = document.getElementById('splitChartWrap');
    const fanPwmSlider = document.getElementById('fanPwmSlider');
    const fanSliderValueEl = document.getElementById('fanSliderValue');
    const fanControlHintEl = document.getElementById('fanControlHint');
    const applyFanPwmBtn = document.getElementById('applyFanPwm');
    const stopFanPwmBtn = document.getElementById('stopFanPwm');

    let fanRequestPending = false;
    let fanSliderDirty = false;

    const metricOrder = ['soil', 'humidity', 'temperature', 'wind', 'light', 'pressure'];
    const metricConfig = {
      soil: {
        label: 'Soil Analog',
        borderColor: '#7b5c2e',
        backgroundColor: 'rgba(123, 92, 46, 0.12)',
        yAxisID: 'ySoil',
      },
      humidity: {
        label: 'Air Humidity (%)',
        borderColor: '#4b78c7',
        backgroundColor: 'rgba(75, 120, 199, 0.1)',
        yAxisID: 'yPercent',
      },
      temperature: {
        label: 'Air Temperature (C)',
        borderColor: '#d14f45',
        backgroundColor: 'rgba(209, 79, 69, 0.1)',
        yAxisID: 'yTemperature',
      },
      wind: {
        label: 'Wind Speed (m/s)',
        borderColor: '#7c4cc9',
        backgroundColor: 'rgba(124, 76, 201, 0.1)',
        yAxisID: 'yWind',
      },
      light: {
        label: 'Light Intensity (lx)',
        borderColor: '#8b6a16',
        backgroundColor: 'rgba(139, 106, 22, 0.12)',
        yAxisID: 'yLux',
      },
      pressure: {
        label: 'Pressure (hPa)',
        borderColor: '#2f8f67',
        backgroundColor: 'rgba(47, 143, 103, 0.1)',
        yAxisID: 'yPressure',
      },
    };

    function createDataset(metricKey) {
      const metric = metricConfig[metricKey];
      return {
        label: metric.label,
        data: [],
        borderColor: metric.borderColor,
        backgroundColor: metric.backgroundColor,
        yAxisID: metric.yAxisID,
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
      };
    }

    function buildScales(activeAxes) {
      const scales = {
        x: {
          grid: { color: '#d6d1c4' },
          ticks: {
            maxTicksLimit: 6,
          }
        }
      };
      if (activeAxes.has('yPercent')) {
        scales.yPercent = {
          type: 'linear',
          position: 'left',
          min: 0,
          max: 100,
          grid: { color: '#d6d1c4' },
          title: { display: true, text: 'Humidity %' },
          ticks: {
            stepSize: 20,
          }
        };
      }
      if (activeAxes.has('ySoil')) {
        scales.ySoil = {
          type: 'linear',
          position: 'left',
          grid: { drawOnChartArea: !activeAxes.has('yPercent') },
          title: { display: true, text: 'Soil ADC' },
          ticks: {
            maxTicksLimit: 6,
          },
          offset: activeAxes.has('yPercent')
        };
      }
      if (activeAxes.has('yTemperature')) {
        scales.yTemperature = {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: !activeAxes.has('yPercent') && !activeAxes.has('ySoil') },
          title: { display: true, text: 'Temperature C' },
          offset: true,
          ticks: {
            stepSize: 2,
          }
        };
      }
      if (activeAxes.has('yLux')) {
        scales.yLux = {
          type: 'linear',
          position: 'left',
          grid: { drawOnChartArea: !activeAxes.has('yPercent') },
          title: { display: true, text: 'Light lx' },
          offset: activeAxes.has('yPercent'),
          ticks: {
            maxTicksLimit: 6,
          }
        };
      }
      if (activeAxes.has('yWind')) {
        scales.yWind = {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: false },
          title: { display: true, text: 'Wind m/s' },
          offset: true,
          ticks: {
            maxTicksLimit: 6,
          }
        };
      }
      if (activeAxes.has('yPressure')) {
        scales.yPressure = {
          type: 'linear',
          position: 'right',
          grid: { drawOnChartArea: false },
          title: { display: true, text: 'Pressure hPa' },
          ticks: {
            stepSize: 5,
          }
        };
      }
      return scales;
    }

    function createLineChart(canvasId, metricKeys) {
      const activeAxes = new Set(metricKeys.map(key => metricConfig[key].yAxisID));
      return new Chart(document.getElementById(canvasId), {
        type: 'line',
        data: {
          labels: [],
          datasets: metricKeys.map(createDataset),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'index', intersect: false },
          scales: buildScales(activeAxes),
          plugins: {
            legend: {
              display: metricKeys.length > 1,
              onClick: () => {},
            },
          },
        }
      });
    }

    function setDatasetVisibility(chart, datasetIndex, visible) {
      chart.setDatasetVisibility(datasetIndex, visible);
      chart.data.datasets[datasetIndex].hidden = !visible;
    }

    const envChart = createLineChart('envChart', metricOrder);
    const soilChart = createLineChart('soilChart', ['soil']);
    const climateChart = createLineChart('climateChart', ['humidity', 'temperature']);
    const lightChart = createLineChart('lightChart', ['light']);
    const windChart = createLineChart('windChart', ['wind']);
    const pressureChart = createLineChart('pressureChart', ['pressure']);

    function activeMetrics() {
      return new Set(metricToggles.filter(toggle => toggle.checked).map(toggle => toggle.dataset.metric));
    }

    function currentChartMode() {
      return chartModeInputs.find(input => input.checked)?.value || 'single';
    }

    function applyChartMode() {
      const mode = currentChartMode();
      singleChartWrap.classList.toggle('hidden', mode !== 'single');
      splitChartWrap.classList.toggle('hidden', mode !== 'split');
    }

    function updateSingleChartVisibility() {
      const enabled = activeMetrics();
      envChart.data.datasets.forEach((dataset, index) => {
        setDatasetVisibility(envChart, index, enabled.has(metricOrder[index]));
      });
      envChart.update();
    }

    function updateSplitChartVisibility() {
      const enabled = activeMetrics();
      document.getElementById('climateChart').closest('.chart-card').classList.toggle(
        'hidden',
        !enabled.has('temperature') && !enabled.has('humidity')
      );
      document.getElementById('soilChart').closest('.chart-card').classList.toggle('hidden', !enabled.has('soil'));
      document.getElementById('windChart').closest('.chart-card').classList.toggle('hidden', !enabled.has('wind'));
      document.getElementById('lightChart').closest('.chart-card').classList.toggle('hidden', !enabled.has('light'));
      document.getElementById('pressureChart').closest('.chart-card').classList.toggle('hidden', !enabled.has('pressure'));

      setDatasetVisibility(soilChart, 0, enabled.has('soil'));
      setDatasetVisibility(climateChart, 0, enabled.has('humidity'));
      setDatasetVisibility(climateChart, 1, enabled.has('temperature'));
      setDatasetVisibility(windChart, 0, enabled.has('wind'));
      setDatasetVisibility(lightChart, 0, enabled.has('light'));
      setDatasetVisibility(pressureChart, 0, enabled.has('pressure'));

      soilChart.update();
      climateChart.update();
      windChart.update();
      lightChart.update();
      pressureChart.update();
    }

    function formatTs(ts) {
      if (!ts) return '--';
      return new Date(ts * 1000).toLocaleTimeString();
    }

    function updateFanSliderLabel(percent) {
      fanSliderValueEl.textContent = `${percent} %`;
    }

    async function sendFanPwm(percent) {
      fanRequestPending = true;
      fanControlHintEl.textContent = `Sending ${percent}%...`;
      applyFanPwmBtn.disabled = true;
      stopFanPwmBtn.disabled = true;

      try {
        const response = await fetch('/api/fan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ percent }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.error || 'fan control failed');
        }
        fanSliderDirty = false;
        fanControlHintEl.textContent = `Fan set to ${payload.percent}%`;
      } catch (error) {
        fanControlHintEl.textContent = `Control failed: ${error.message}`;
      } finally {
        fanRequestPending = false;
        applyFanPwmBtn.disabled = false;
        stopFanPwmBtn.disabled = false;
      }
    }

    metricToggles.forEach(toggle => toggle.addEventListener('change', () => {
      updateSingleChartVisibility();
      updateSplitChartVisibility();
    }));
    chartModeInputs.forEach(input => input.addEventListener('change', applyChartMode));
    fanPwmSlider.addEventListener('input', () => {
      fanSliderDirty = true;
      updateFanSliderLabel(fanPwmSlider.value);
    });
    applyFanPwmBtn.addEventListener('click', () => sendFanPwm(Number(fanPwmSlider.value)));
    stopFanPwmBtn.addEventListener('click', () => {
      fanPwmSlider.value = '0';
      updateFanSliderLabel(0);
      sendFanPwm(0);
    });

    async function refresh() {
      const [latestRes, historyRes, statusRes] = await Promise.all([
        fetch('/api/latest'),
        fetch('/api/history'),
        fetch('/api/status')
      ]);

      const latest = await latestRes.json();
      const hist = await historyRes.json();
      const status = await statusRes.json();

      statusEl.textContent = `BLE status: ${status.state} | ${status.message}`;

      if (latest && latest.ts !== undefined) {
        airTemperatureEl.textContent = latest.air_temperature === null
          ? '-- C'
          : `${latest.air_temperature.toFixed(1)} C`;
        airHumidityEl.textContent = latest.air_humidity === null
          ? '-- %'
          : `${latest.air_humidity.toFixed(0)} %`;
        soilHumidityEl.textContent = latest.soil_humidity === null
          ? '--'
          : `${latest.soil_humidity}`;
        windSpeedEl.textContent = latest.wind_speed === null
          ? '-- m/s'
          : `${latest.wind_speed.toFixed(2)} m/s`;
        lightLuxEl.textContent = latest.light_lux === null
          ? '-- lx'
          : `${latest.light_lux.toFixed(0)} lx`;
        pressureHpaEl.textContent = latest.pressure_hpa === null
          ? '-- hPa'
          : `${latest.pressure_hpa.toFixed(1)} hPa`;
        fanPwmValueEl.textContent = latest.fan_pwm_percent === null || latest.fan_pwm_percent === undefined
          ? '-- %'
          : `${latest.fan_pwm_percent} %`;
        if (!fanSliderDirty && !fanRequestPending && latest.fan_pwm_percent !== null && latest.fan_pwm_percent !== undefined) {
          fanPwmSlider.value = `${latest.fan_pwm_percent}`;
          updateFanSliderLabel(latest.fan_pwm_percent);
        }
        sensorStateEl.textContent = `Soil ADC ${latest.soil_humidity ?? '--'} | Fan ${latest.fan_pwm_percent ?? '--'}% | Wind ${latest.wind} | AHT20 ${latest.aht20} | BH1750 ${latest.bh1750} | BMP280 ${latest.bmp280}`;
        packetTimeEl.textContent = formatTs(latest.ts);
      }

      const labels = hist.map(point => new Date(point.ts * 1000).toLocaleTimeString());
      const soilSeries = hist.map(point => point.soil_humidity);
      const humiditySeries = hist.map(point => point.air_humidity);
      const temperatureSeries = hist.map(point => point.air_temperature);
      const windSeries = hist.map(point => point.wind_speed);
      const lightSeries = hist.map(point => point.light_lux);
      const pressureSeries = hist.map(point => point.pressure_hpa);

      envChart.data.labels = labels;
      envChart.data.datasets[0].data = soilSeries;
      envChart.data.datasets[1].data = humiditySeries;
      envChart.data.datasets[2].data = temperatureSeries;
      envChart.data.datasets[3].data = windSeries;
      envChart.data.datasets[4].data = lightSeries;
      envChart.data.datasets[5].data = pressureSeries;
      envChart.update();

      soilChart.data.labels = labels;
      soilChart.data.datasets[0].data = soilSeries;
      soilChart.update();

      climateChart.data.labels = labels;
      climateChart.data.datasets[0].data = humiditySeries;
      climateChart.data.datasets[1].data = temperatureSeries;
      climateChart.update();

      windChart.data.labels = labels;
      windChart.data.datasets[0].data = windSeries;
      windChart.update();

      lightChart.data.labels = labels;
      lightChart.data.datasets[0].data = lightSeries;
      lightChart.update();

      pressureChart.data.labels = labels;
      pressureChart.data.datasets[0].data = pressureSeries;
      pressureChart.update();
    }

    applyChartMode();
    updateSingleChartVisibility();
    updateSplitChartVisibility();
    updateFanSliderLabel(0);
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


def update_status(state: str, message: str) -> None:
    with state_lock:
        ble_status["state"] = state
        ble_status["message"] = message
        ble_status["last_update"] = time.time()


def handle_notification(_: int, data: bytearray) -> None:
  text = data.decode("utf-8", errors="replace")
  try:
    payload = json.loads(text)
    if isinstance(payload, list):
      if len(payload) >= 7:
        soil_humidity = payload[0] if len(payload) > 0 else None
        air_temperature = payload[1] if len(payload) > 1 else None
        air_humidity = payload[2] if len(payload) > 2 else None
        wind_speed = payload[3] if len(payload) > 3 else None
        light_lux = payload[4] if len(payload) > 4 else None
        pressure_hpa = payload[5] if len(payload) > 5 else None
        fan_pwm_percent = payload[6] if len(payload) > 6 else 0
      elif len(payload) >= 6:
        soil_humidity = payload[0] if len(payload) > 0 else None
        air_temperature = payload[1] if len(payload) > 1 else None
        air_humidity = payload[2] if len(payload) > 2 else None
        wind_speed = payload[3] if len(payload) > 3 else None
        light_lux = payload[4] if len(payload) > 4 else None
        pressure_hpa = payload[5] if len(payload) > 5 else None
        fan_pwm_percent = 0
      elif len(payload) >= 5:
        soil_humidity = payload[0] if len(payload) > 0 else None
        air_temperature = payload[1] if len(payload) > 1 else None
        air_humidity = payload[2] if len(payload) > 2 else None
        wind_speed = None
        light_lux = payload[3] if len(payload) > 3 else None
        pressure_hpa = payload[4] if len(payload) > 4 else None
        fan_pwm_percent = 0
      else:
        soil_humidity = None
        air_temperature = payload[0] if len(payload) > 0 else None
        air_humidity = payload[1] if len(payload) > 1 else None
        wind_speed = None
        light_lux = payload[2] if len(payload) > 2 else None
        pressure_hpa = payload[3] if len(payload) > 3 else None
        fan_pwm_percent = 0

      millis = 0
      soil_state = str(int(soil_humidity)) if soil_humidity is not None else "offline"
      wind_state = "online" if wind_speed is not None else "offline"
      aht20_state = "online" if air_temperature is not None else "offline"
      bh1750_state = "online" if light_lux is not None else "offline"
      bmp280_state = "online" if pressure_hpa is not None else "offline"
    else:
      soil_humidity = payload.get("soilHumidity")
      air_temperature = payload.get("airTemperature")
      air_humidity = payload.get("airHumidity")
      wind_speed = payload.get("windSpeed")
      light_lux = payload.get("lightLux")
      pressure_hpa = payload.get("pressureHpa")
      fan_pwm_percent = payload.get("fanPwmPercent", 0)
      millis = int(payload.get("millis", 0))
      soil_state = payload.get("soilState", str(int(soil_humidity)) if soil_humidity is not None else "offline")
      wind_state = payload.get("wind", "online" if wind_speed is not None else "offline")
      aht20_state = payload.get("aht20", "online" if air_temperature is not None else "offline")
      bh1750_state = payload.get("bh1750", "online" if light_lux is not None else "offline")
      bmp280_state = payload.get("bmp280", "online" if pressure_hpa is not None else "offline")

    point = {
        "ts": time.time(),
        "soil_humidity": int(soil_humidity) if soil_humidity is not None else None,
        "soil_state": soil_state,
        "air_temperature": float(air_temperature) if air_temperature is not None else None,
        "air_humidity": float(air_humidity) if air_humidity is not None else None,
        "wind_speed": float(wind_speed) if wind_speed is not None else None,
        "light_lux": float(light_lux) if light_lux is not None else None,
        "pressure_hpa": float(pressure_hpa) if pressure_hpa is not None else None,
        "fan_pwm_percent": int(fan_pwm_percent) if fan_pwm_percent is not None else 0,
        "wind": wind_state,
        "aht20": aht20_state,
        "bh1750": bh1750_state,
        "bmp280": bmp280_state,
        "millis": millis,
    }
    with state_lock:
        history.append(point)
  except (ValueError, json.JSONDecodeError):
    update_status("warn", f"invalid payload: {text[:50]}")


async def write_fan_pwm(percent: int) -> None:
    global desired_fan_percent

    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")

    client = ble_client
    if client is None or not client.is_connected:
        raise RuntimeError("BLE device not connected")

    await client.write_gatt_char(FAN_CHAR_UUID, str(percent).encode("utf-8"), response=True)
    desired_fan_percent = percent

    with state_lock:
        if history:
            history[-1]["fan_pwm_percent"] = percent


async def ble_worker() -> None:
    global ble_client

    while True:
        try:
            update_status("scan", f"searching {DEVICE_NAME}")
            device = await BleakScanner.find_device_by_filter(
                lambda d, adv: d.name == DEVICE_NAME
                or (
                    adv is not None
                    and SERVICE_UUID.lower() in [uuid.lower() for uuid in (adv.service_uuids or [])]
                ),
                timeout=10.0,
            )
            if device is None:
                update_status("scan", "device not found, retrying")
                await asyncio.sleep(2.0)
                continue

            update_status("connect", f"connecting {device.address}")
            async with BleakClient(device) as client:
                ble_client = client
                await client.start_notify(CHAR_UUID, handle_notification)
                await write_fan_pwm(desired_fan_percent)
                update_status("connected", "receiving notifications")

                while client.is_connected:
                    await asyncio.sleep(1.0)

                ble_client = None
                update_status("scan", "disconnected, retrying")
        except Exception as exc:  # pylint: disable=broad-except
            ble_client = None
            update_status("error", str(exc))
            await asyncio.sleep(2.0)


def run_ble_loop() -> None:
    global ble_loop

    ble_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ble_loop)
    ble_loop.run_until_complete(ble_worker())


@app.get("/")
def index() -> str:
    return render_template_string(DASHBOARD_HTML)


@app.get("/api/latest")
def api_latest() -> Any:
    with state_lock:
        if history:
            return jsonify(history[-1])
    return jsonify({})


@app.get("/api/history")
def api_history() -> Any:
    with state_lock:
        return jsonify(list(history))


@app.get("/api/status")
def api_status() -> Any:
    with state_lock:
        return jsonify(ble_status)


@app.post("/api/fan")
def api_fan() -> Any:
    payload = request.get_json(silent=True) or {}
    percent = payload.get("percent")

    try:
        percent_value = int(percent)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "percent must be an integer"}), 400

    if percent_value < 0 or percent_value > 100:
        return jsonify({"ok": False, "error": "percent must be between 0 and 100"}), 400

    if ble_loop is None:
        return jsonify({"ok": False, "error": "BLE loop not ready"}), 503

    future = asyncio.run_coroutine_threadsafe(write_fan_pwm(percent_value), ble_loop)
    try:
        future.result(timeout=5)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"ok": False, "error": str(exc)}), 503

    update_status("connected", f"fan set to {percent_value}%")
    return jsonify({"ok": True, "percent": percent_value})


def main() -> None:
    parser = argparse.ArgumentParser(description="ESP32 BLE environmental dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--no-browser", action="store_true", help="do not auto-open browser")
    args = parser.parse_args()

    t = threading.Thread(target=run_ble_loop, daemon=True)
    t.start()

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"Dashboard started: {url}")
    print("If dependencies are missing, install: pip install flask bleak")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
