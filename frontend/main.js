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
const resetStatsBtn = document.getElementById("resetStatsBtn");
const resetStatsFeedback = document.getElementById("resetStatsFeedback");

const MAX_TRADES_TO_DISPLAY = 50;
const METRIC_DISPLAY_LIMIT = 12;
const LIVE_METRIC_FIELDS = [
  { key: "markets", label: "Markets" },
  { key: "paper_balance", label: "Paper Balance" },
  { key: "open_positions", label: "Open Positions" },
  { key: "total_realized_pnl", label: "Realized PnL" },
  { key: "win_rate", label: "Win Rate", formatter: (value) => formatPercent(value) },
  { key: "profit_factor", label: "Profit Factor" },
  { key: "expectancy", label: "Expectancy" },
  { key: "sharpe", label: "Sharpe" },
];

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0.00";
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return num > 0 ? "∞" : num < 0 ? "-∞" : "0";
  }
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

function renderLiveMetrics(metrics) {
  if (!metrics) {
    renderMetricList(liveMetricsList, null, "Start demo-live to stream live metrics.");
    return;
  }

  liveMetricsList.innerHTML = "";
  LIVE_METRIC_FIELDS.forEach(({ key, label, formatter }) => {
    const value = metrics[key];
    if (value === undefined) {
      return;
    }
    const li = document.createElement("li");
    const formatted = formatter ? formatter(value) : formatNumber(value);
    li.innerHTML = `<strong>${label}</strong>: ${formatted}`;
    liveMetricsList.appendChild(li);
  });
}

function deriveWinRateFromSummaries(summaries) {
  if (!summaries || summaries.length === 0) {
    return 0;
  }
  const totals = summaries.reduce(
    (acc, entry) => {
      const trades = Number(entry.trades ?? 0);
      const wins = trades * Number(entry.win_rate ?? 0);
      return { trades: acc.trades + trades, wins: acc.wins + wins };
    },
    { trades: 0, wins: 0 }
  );
  if (totals.trades === 0) {
    return 0;
  }
  return totals.wins / totals.trades;
}

function buildLiveMetrics(status) {
  const hasMetrics = (payload) => payload && Object.keys(payload).length > 0;
  const source = hasMetrics(status.live_metrics) ? status.live_metrics : hasMetrics(status.metrics) ? status.metrics : null;
  if (source) {
    return {
      markets: source.markets ?? (status.symbol_summaries?.length ?? 0),
      paper_balance: source.paper_balance ?? status.balance ?? 0,
      open_positions: source.open_positions ?? (status.open_positions?.length ?? 0),
      total_realized_pnl: source.total_realized_pnl ?? status.realized_pnl ?? 0,
      win_rate: source.win_rate ?? deriveWinRateFromSummaries(status.symbol_summaries),
      profit_factor: source.profit_factor ?? 0,
      expectancy: source.expectancy ?? 0,
      sharpe: source.sharpe ?? 0,
    };
  }

  const summaries = status.symbol_summaries ?? [];
  const derived = summaries.reduce(
    (acc, entry) => {
      const trades = Number(entry.trades ?? 0);
      const realized = Number(entry.realized_pnl ?? 0);
      const openPnl = Number(entry.open_pnl ?? 0);
      const winRate = Number(entry.win_rate ?? 0);
      acc.trades += trades;
      acc.winPoints += trades * winRate;
      acc.realized += realized;
      acc.openPnL += openPnl;
      return acc;
    },
    { trades: 0, winPoints: 0, realized: 0, openPnL: 0 }
  );
  const recent = status.recent_trades ?? [];
  const pnlWins = recent.filter((trade) => Number(trade.pnl ?? 0) > 0).map((trade) => Number(trade.pnl ?? 0));
  const pnlLosses = recent.filter((trade) => Number(trade.pnl ?? 0) <= 0).map((trade) => Number(trade.pnl ?? 0));
  const lossSum = pnlLosses.reduce((sum, value) => sum + value, 0);
  const profitFactor = lossSum !== 0 ? (pnlWins.reduce((sum, value) => sum + value, 0) / Math.abs(lossSum)) : pnlWins.length ? Infinity : 0;
  const winRate = derived.trades ? derived.winPoints / derived.trades : 0;
  const expectancy = derived.trades ? derived.realized / derived.trades : 0;

  return {
    markets: summaries.length,
    paper_balance: status.balance ?? 0,
    open_positions: status.open_positions?.length ?? 0,
    total_realized_pnl: derived.realized,
    win_rate: winRate,
    profit_factor: profitFactor,
    expectancy,
    sharpe: 0,
  };
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
  renderLiveMetrics(buildLiveMetrics(status));
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

  async function handleResetStats() {
    if (!resetStatsBtn) {
      return;
    }
    resetStatsBtn.disabled = true;
    if (resetStatsFeedback) {
      resetStatsFeedback.textContent = "Resetting stats...";
    }
    try {
      const response = await fetch("/api/reset-stats", { method: "POST" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      await fetchInitialStatus();
      if (resetStatsFeedback) {
        const timestamp = payload.updated_at ? new Date(payload.updated_at * 1000).toLocaleTimeString() : new Date().toLocaleTimeString();
        resetStatsFeedback.textContent = `Stats reset at ${timestamp}`;
      }
    } catch (error) {
      console.warn("Failed to reset stats", error);
      if (resetStatsFeedback) {
        resetStatsFeedback.textContent = `Reset failed: ${error.message}`;
      }
    } finally {
      resetStatsBtn.disabled = false;
    }
  }
  ws.addEventListener("close", () => {
    setConnectionStatus("reconnecting");
    setTimeout(connectWebSocket, 750);
  });
  if (resetStatsBtn) {
    resetStatsBtn.addEventListener("click", handleResetStats);
  }
  ws.addEventListener("error", () => {
    setConnectionStatus("disconnected");
    ws.close();
  });
}

setConnectionStatus("disconnected");
fetchInitialStatus();
connectWebSocket();
