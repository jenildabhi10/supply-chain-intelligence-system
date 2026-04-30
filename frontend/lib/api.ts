// Typed client for the Supply Chain Intelligence FastAPI backend.
// Endpoints mirror src/api/main.py.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export type RiskTier = "low" | "medium" | "high" | "critical";

export interface PortInfo {
  port_id: string;
  name:    string;
  region:  string;
  state:   string;
  lat:     number;
  lon:     number;
}

export interface PortListResponse {
  ports: PortInfo[];
  total: number;
}

export interface RiskSummary {
  port_id:          string;
  port_name:        string;
  risk_score:       number;
  risk_tier:        RiskTier;
  p_disruption:     number;
  anomaly_score:    number;
  confidence_score: number;
  n_signals:        number;
  top_driver:       string | null;
  generated_at:     string;
}

export interface AllRiskResponse {
  port_summaries: RiskSummary[];
  total_ports:    number;
  generated_at:   string;
}

export interface SHAPDriver {
  feature:     string;
  value:       number;
  shap_impact: number;
  direction:   "increases_risk" | "decreases_risk";
}

export interface ActiveSignal {
  signal_id:    string;
  source:       "nws" | "gdelt" | "usgs" | "firms" | "bts";
  signal_type:  string;
  description:  string;
  severity:     "low" | "medium" | "high" | "critical";
  port_id:      string;
  observed_at:  string;
  evidence_url: string | null;
}

export interface ModelInfo {
  version:         string;
  trained_at:      string | null;
  cv_pr_auc:       number | null;
  n_training_rows: number | null;
}

export interface RiskCard {
  card_id:               string;
  port_id:               string;
  port_name:             string;
  region:                string;
  risk_score:            number;
  risk_tier:             RiskTier;
  p_disruption:          number;
  anomaly_score:         number;
  impact_index:          number;
  confidence_score:      number;
  confidence_flags:      string[];
  data_completeness:     number;
  forecast_horizon_days: number;
  features_week_ending:  string | null;
  top_shap_drivers:      SHAPDriver[];
  active_signals:        ActiveSignal[];
  recommended_actions:   string[];
  model_info:            ModelInfo | null;
  generated_at:          string;
}

export interface AlertsResponse {
  port_id:       string;
  port_name:     string;
  lookback_days: number;
  signal_count:  number;
  signals:       ActiveSignal[];
}

export interface ExplainResponse {
  answer:          string;
  tool_calls_made: string[];
  llm_used:        boolean;
  model:           string | null;
  question:        string;
}

export interface SimulateResponse {
  port_id:             string;
  port_name:           string;
  scenario:            { p_disruption: number; anomaly_score: number; impact_index: number };
  risk_score:          number;
  risk_tier:           RiskTier;
  delta_score:         number;
  baseline_score:      number;
  recommended_actions: string[];
}

export interface HealthResponse {
  status:        string;
  model_loaded:  boolean;
  llm_available: boolean;
  timestamp:     string;
}

// ─── Fetch helpers ────────────────────────────────────────────────────────────

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`GET ${path} failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`POST ${path} failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

export const api = {
  health:  () => get<HealthResponse>("/health"),
  ports:   () => get<PortListResponse>("/ports"),
  allRisk: () => get<AllRiskResponse>("/risk"),
  portRisk: (id: string) => get<RiskCard>(`/risk/${id}`),
  alerts:  (id: string, lookbackDays = 7) =>
    get<AlertsResponse>(`/alerts/${id}?lookback_days=${lookbackDays}`),
  explain: (question: string) =>
    post<ExplainResponse>("/explain", { question }),
  simulate: (port_id: string, p_disruption: number, anomaly_score: number) =>
    post<SimulateResponse>("/simulate", { port_id, p_disruption, anomaly_score }),
};
