const STATE_URL = "/api/dashboard/state";
const EVENT_LIMIT = 25;
const REFRESH_MS = 5000;

const equityValueEl = document.getElementById("equityValue");
const balanceValueEl = document.getElementById("balanceValue");
const unrealizedValueEl = document.getElementById("unrealizedValue");
const openPositionsValueEl = document.getElementById("openPositionsValue");
const updatedAtEl = document.getElementById("updatedAt");
const positionsBodyEl = document.getElementById("positionsBody");
const eventsListEl = document.getElementById("eventsList");

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0";
  }
  const num = Number(value);
  return Math.abs(num) >= 1000 ? num.toLocaleString(undefined, { maximumFractionDigits: 2 }) : num.toFixed(2);
}

function safeDate(value) {
  if (!value) {
    return "No updates yet";
  }
  try {
    return new Date(value).toLocaleString();
  } catch (error) {
    return value;
  }
}

function renderSnapshot(equity) {
  equityValueEl.textContent = formatNumber(equity?.equity);
  balanceValueEl.textContent = formatNumber(equity?.balance);
  unrealizedValueEl.textContent = formatNumber(equity?.unrealized_pnl);
  openPositionsValueEl.textContent = equity?.open_positions ?? 0;
  updatedAtEl.textContent = `Updated ${safeDate(equity?.timestamp || equity?.updated_at)}`;
}

function renderPositions(positions) {
  positionsBodyEl.innerHTML = "";
  if (!positions || positions.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.textContent = "No open positions";
    row.appendChild(cell);
    positionsBodyEl.appendChild(row);
    return;
  }

  positions.forEach((pos) => {
    const row = document.createElement("tr");
    const cols = [
      pos.symbol ?? "—",
      pos.side ?? "—",
      formatNumber(pos.quantity),
      formatNumber(pos.entry_price),
      pos.stop_loss ? formatNumber(pos.stop_loss) : "—",
      pos.take_profit ? formatNumber(pos.take_profit) : "—",
    ];
    cols.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    positionsBodyEl.appendChild(row);
  });
}

function renderEvents(events) {
  eventsListEl.innerHTML = "";
  if (!events || events.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No events yet";
    eventsListEl.appendChild(li);
    return;
  }

  events.slice(0, 25).forEach((event) => {
    const li = document.createElement("li");
    const details = [];
    if (event.symbol) details.push(event.symbol);
    if (event.side) details.push(event.side);
    if (event.event) details.push(`• ${event.event}`);
    if (event.qty) details.push(`qty ${formatNumber(event.qty)}`);
    if (event.fill_price) details.push(`@ ${formatNumber(event.fill_price)}`);
    if (event.pnl) details.push(`PnL ${formatNumber(event.pnl)}`);
    li.innerHTML = `<strong>${details.join(" ")}</strong><br /><span>${safeDate(event.timestamp)}</span>`;
    eventsListEl.appendChild(li);
  });
}

async function fetchState() {
  try {
    const response = await fetch(`${STATE_URL}?limit=${EVENT_LIMIT}&t=${Date.now()}`);
    if (!response.ok) {
      throw new Error(`Telemetry fetch failed with status ${response.status}`);
    }
    const payload = await response.json();
    renderSnapshot(payload.equity || payload);
    renderPositions(payload.positions || []);
    renderEvents(payload.recent_events || []);
  } catch (error) {
    updatedAtEl.textContent = `Telemetry unavailable: ${error.message}`;
    updatedAtEl.classList.add("error");
    renderPositions([]);
    eventsListEl.innerHTML = "<li>Waiting for telemetry...</li>";
  }
}

fetchState();
setInterval(fetchState, REFRESH_MS);
