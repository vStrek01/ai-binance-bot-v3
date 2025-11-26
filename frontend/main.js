const connectionEl = document.getElementById("connectionStatus");
const modeEl = document.getElementById("mode");
const symbolInfoEl = document.getElementById("symbolInfo");
const updatedAtEl = document.getElementById("updatedAt");
const balanceEl = document.getElementById("balance");
const equityEl = document.getElementById("equity");
const pnlEl = document.getElementById("pnl");
const progressEl = document.getElementById("progress");
const positionsBody = document.getElementById("positionsBody");
const symbolStatsBody = document.getElementById("symbolStatsBody");
const tradesList = document.getElementById("tradesList");
const metricsList = document.getElementById("metricsList");
const liveMetricsList = document.getElementById("liveMetricsList");
const portfolioLabelEl = document.getElementById("portfolioLabel");
const portfolioMetricEl = document.getElementById("portfolioMetric");
const portfolioList = document.getElementById("portfolioList");

const MAX_TRADES_TO_DISPLAY = 50;
const METRIC_DISPLAY_LIMIT = 12;

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0.00";
  }
  const num = Number(value);
  if (Math.abs(num) >= 1000) {
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return num.toFixed(2);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0.0%";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatProgress(progress) {
  if (!progress || !progress.total) {
    return "—";
  }
  return `${progress.completed ?? 0} / ${progress.total}`;
}

function setConnectionStatus(state) {
  connectionEl.classList.remove("connected", "reconnecting", "disconnected");
  connectionEl.classList.add(state);
  const label = state === "connected" ? "Connected" : state === "reconnecting" ? "Reconnecting" : "Disconnected";
  connectionEl.textContent = label;
}

function createCell(value, options = {}) {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (options.align === "right") {
    cell.classList.add("numeric");
  }
  if (options.className) {
    cell.classList.add(options.className);
  }
  if (options.title) {
    cell.title = options.title;
  }
  return cell;
}

function renderPositions(positions) {
  positionsBody.innerHTML = "";
  if (!positions || positions.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 9;
    cell.textContent = "No open positions";
    row.appendChild(cell);
    positionsBody.appendChild(row);
    return;
  }

  positions.forEach((pos) => {
    const row = document.createElement("tr");
    const volatility = pos.volatility ?? {};
    const signals = pos.signals ?? {};
    const cells = [
      { value: pos.symbol ?? "—" },
      { value: pos.side ?? "—" },
      { value: formatNumber(pos.quantity), align: "right" },
      { value: formatNumber(pos.entry_price), align: "right" },
      { value: formatNumber(pos.mark_price), align: "right" },
      { value: formatNumber(pos.pnl), align: "right" },
      { value: pos.sizing_mode ?? "—" },
      {
        value: `ATR ${formatNumber(volatility.atr ?? 0)} / σ ${formatNumber(volatility.stddev ?? 0)}`,
        className: "wrap",
      },
      {
        value: signals.regime ? `${signals.regime} (${formatNumber(signals.score ?? 0)})` : "—",
        className: "wrap",
      },
    ];
    cells.forEach((config) => row.appendChild(createCell(config.value, config)));
    positionsBody.appendChild(row);
  });
}

function renderSymbolStats(entries) {
  symbolStatsBody.innerHTML = "";
  if (!entries || entries.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.textContent = "No symbols tracked";
    row.appendChild(cell);
    symbolStatsBody.appendChild(row);
    return;
  }

  entries.forEach((entry) => {
    const row = document.createElement("tr");
    const cells = [
      { value: entry.symbol ?? "—" },
      { value: entry.timeframe ?? "—" },
      { value: entry.trades ?? 0, align: "right" },
      { value: formatPercent(entry.win_rate), align: "right" },
      { value: formatNumber(entry.realized_pnl), align: "right" },
      { value: formatNumber(entry.open_pnl), align: "right" },
    ];
    cells.forEach((config) => row.appendChild(createCell(config.value, config)));
    symbolStatsBody.appendChild(row);
  });
}

function renderTrades(trades) {
  tradesList.innerHTML = "";
  if (!trades || trades.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No trades yet";
    tradesList.appendChild(li);
    return;
  }

  trades.slice(0, MAX_TRADES_TO_DISPLAY).forEach((trade) => {
    const li = document.createElement("li");
    const label = [
      trade.mode ?? "",
      trade.symbol ?? "",
      trade.timeframe ? `(${trade.timeframe})` : "",
      trade.side ? `• ${trade.side}` : "",
      `• PnL ${formatNumber(trade.pnl)}`,
    ]
      .filter(Boolean)
      .join(" ");
    li.innerHTML = `<strong>${label}</strong>`;
    li.title = `Entry ${formatNumber(trade.entry_price)} → Exit ${formatNumber(trade.exit_price)} | Reason ${
      trade.reason ?? ""
    }`;
    tradesList.appendChild(li);
  });
}

function renderMetricList(targetEl, metrics, emptyMessage) {
  targetEl.innerHTML = "";
  if (!metrics || Object.keys(metrics).length === 0) {
    const li = document.createElement("li");
    li.textContent = emptyMessage;
    targetEl.appendChild(li);
    return;
  }

  Object.entries(metrics)
    .slice(0, METRIC_DISPLAY_LIMIT)
    .forEach(([key, value]) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${key}</strong>: ${formatNumber(value)}`;
      targetEl.appendChild(li);
    });
}

function renderPortfolio(snapshot) {
  if (!snapshot) {
    portfolioLabelEl.textContent = "—";
    portfolioMetricEl.textContent = "—";
    portfolioList.innerHTML = "<li>No portfolio selection yet</li>";
    return;
  }

  const symbols = snapshot.symbols ?? [];
  portfolioLabelEl.textContent = snapshot.label ?? "—";
  portfolioMetricEl.textContent = snapshot.metric ?? "—";
  portfolioList.innerHTML = "";

  if (symbols.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No portfolio selection yet";
    portfolioList.appendChild(li);
    return;
  }

  symbols.forEach((entry) => {
    const li = document.createElement("li");
    const metricValue = entry.metric ?? entry.total_pnl ?? 0;
    const profitFactor = entry.profit_factor ? ` • PF ${formatNumber(entry.profit_factor)}` : "";
    li.innerHTML = `<strong>${entry.symbol ?? "?"}</strong> ${entry.timeframe ?? ""} • ${formatNumber(
      metricValue
    )}${profitFactor}`;
    portfolioList.appendChild(li);
  });
}

function updateUi(status) {
  modeEl.textContent = status.mode ?? "idle";
  symbolInfoEl.textContent = status.symbol && status.timeframe ? `${status.symbol} / ${status.timeframe}` : "—";
  updatedAtEl.textContent = status.updated_at
    ? new Date(status.updated_at * 1000).toLocaleTimeString()
    : new Date().toLocaleTimeString();
  balanceEl.textContent = formatNumber(status.balance);
  equityEl.textContent = formatNumber(status.equity);
  pnlEl.textContent = `${formatNumber(status.open_pnl)} / ${formatNumber(status.realized_pnl)}`;
  progressEl.textContent = formatProgress(status.progress);

  renderPositions(status.open_positions);
  renderSymbolStats(status.symbol_summaries);
  renderTrades(status.recent_trades);
  renderMetricList(metricsList, status.metrics, "Run a backtest or dry-run to populate metrics.");
  renderMetricList(liveMetricsList, status.live_metrics, "Start demo-live to stream live metrics.");
  renderPortfolio(status.portfolio);
}

async function fetchInitialStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) {
      throw new Error(`Failed to fetch status: ${response.status}`);
    }
    const payload = await response.json();
    updateUi(payload);
  } catch (error) {
    console.warn("Unable to fetch initial status", error);
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/status`);

  ws.addEventListener("open", () => setConnectionStatus("connected"));
  ws.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      updateUi(payload);
    } catch (error) {
      console.warn("Failed to parse websocket payload", error);
    }
  });
  ws.addEventListener("close", () => {
    setConnectionStatus("reconnecting");
    setTimeout(connectWebSocket, 1500);
  });
  ws.addEventListener("error", () => {
    setConnectionStatus("disconnected");
    ws.close();
  });
}

setConnectionStatus("disconnected");
fetchInitialStatus();
connectWebSocket();
