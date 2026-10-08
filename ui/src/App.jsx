import { useMemo, useState } from "react";

/* ------------------------------------------------------------------
   Trained Model Coefficients (Dual-Stage System)
   Post-Midterm (Weeks 7+):
     final_score = 1.77 + 0.1907*att + 0.3968*study + 0.1882*assign + 0.3011*midterm + 4.3972*gpa
   Pre-Midterm (Weeks 1-6):
     final_score = 1.18 + 0.2446*att + 0.6181*study + 0.2911*assign + 6.6735*gpa
------------------------------------------------------------------ */
const FULL_INTERCEPT = 1.7715;
const EARLY_INTERCEPT = 1.1797;
const FULL_MAE = 2.6;
const EARLY_MAE = 3.0;

const ALL_FIELDS = [
  { key: "midterm", label: "Midterm exam score", unit: "pts (0-100)", min: 0, max: 100, step: 1, hint: "Primary mid-semester signal (slope: +0.30 pts)", fullCoef: 0.3011, earlyCoef: 0.0, mean: 79.21, isMidterm: true },
  { key: "study", label: "Dedicated study hours", unit: "hrs / week", min: 0, max: 60, step: 0.5, hint: "Weekly self-study volume", fullCoef: 0.3968, earlyCoef: 0.6181, mean: 21.30 },
  { key: "gpa", label: "Prior cumulative GPA", unit: "pts (0.0-4.0)", min: 0, max: 4.0, step: 0.05, hint: "Historical academic foundation", fullCoef: 4.3972, earlyCoef: 6.6735, mean: 3.01 },
  { key: "assignment", label: "Assignment average", unit: "pts (0-100)", min: 0, max: 100, step: 1, hint: "Continuous formative coursework", fullCoef: 0.1882, earlyCoef: 0.2911, mean: 85.06 },
  { key: "attendance", label: "Classroom attendance", unit: "% (0-100)", min: 0, max: 100, step: 1, hint: "Lecture presence & participation", fullCoef: 0.1907, earlyCoef: 0.2446, mean: 77.85 },
];

const PRESETS = {
  "Average student": { attendance: "78", study: "21", assignment: "85", midterm: "79", gpa: "3.00" },
  "High performer": { attendance: "96", study: "30", assignment: "94", midterm: "90", gpa: "3.80" },
  "At risk": { attendance: "55", study: "10", assignment: "58", midterm: "45", gpa: "2.30" },
  "Test anxiety (bad exam day)": { attendance: "92", study: "26", assignment: "90", midterm: "44", gpa: "3.65" }
};

const clamp = (n, min = 0, max = 100) => Math.max(min, Math.min(max, n));

function errorFor(f, raw) {
  if (raw === "") return "Enter a value.";
  const n = Number(raw);
  if (Number.isNaN(n)) return "Use numbers only.";
  if (n < f.min || n > f.max) return `Enter a value from ${f.min} to ${f.max}.`;
  return null;
}

function predict(v, isEarlyStage) {
  let score = isEarlyStage ? EARLY_INTERCEPT : FULL_INTERCEPT;
  const activeFields = isEarlyStage ? ALL_FIELDS.filter((f) => !f.isMidterm) : ALL_FIELDS;

  const parts = activeFields.map((f) => {
    const x = clamp(v[f.key], f.min, f.max);
    const coef = isEarlyStage ? f.earlyCoef : f.fullCoef;
    score += coef * x;
    return {
      key: f.key,
      label: f.label,
      delta: coef * (x - f.mean),
    };
  });
  return { score: clamp(score, 0, 100), parts };
}

function checkAnomaly(v) {
  const expected = 0.55 * v.assignment + 0.35 * (v.study * 1.5) + v.gpa * 11.0;
  const clampedExpected = clamp(expected, 20, 100);
  const deficit = clampedExpected - v.midterm;
  return {
    isAnomaly: deficit >= 20.0 && v.assignment >= 75.0,
    expected: Math.round(clampedExpected),
    deficit: Math.round(deficit),
  };
}

function advice(v, isEarlyStage) {
  const t = [];
  if (!isEarlyStage && v.midterm < 65) {
    t.push([
      "Review core midterm syllabus",
      `At ${v.midterm} points, midterm recovery is the most direct academic signal. Target units with the highest mark deductions.`,
    ]);
  }
  if (v.attendance < 75) {
    t.push([
      "Increase lecture attendance",
      `At ${v.attendance}% attendance, missed class discussions reduce retention. Raising attendance to 85% or higher recovers approximately +1.9 to +2.4 points per 10%.`,
    ]);
  }
  if (v.study < 15) {
    t.push([
      "Expand weekly study allocation",
      `${v.study} hours/week is below the cohort benchmark. Adding 5 structured study hours outside class is associated with a +2.0 to +3.1 point gain.`,
    ]);
  }
  if (v.assignment < 70) {
    t.push([
      "Target assignment completion",
      "Formative homework performance directly reinforces final exam retention. Request feedback on low-scoring submissions.",
    ]);
  }
  if (v.gpa < 2.5) {
    t.push([
      "Establish structured tutoring",
      "Cumulative GPA indicates foundational concepts may need reinforcement. Schedule weekly advisor check-ins early in the term.",
    ]);
  }
  if (!t.length) {
    t.push([
      "Maintain current academic routine",
      "All pre-final metrics are tracking above cohort benchmarks. Maintain current study and attendance cadence through final review week.",
    ]);
  }
  return t;
}

const band = (s) =>
  s >= 75
    ? { name: "Low Risk (On Track)", text: "text-emerald-300", stroke: "stroke-emerald-400", bg: "bg-emerald-400" }
    : s >= 60
    ? { name: "Moderate Risk (Watchlist)", text: "text-amber-300", stroke: "stroke-amber-400", bg: "bg-amber-400" }
    : { name: "High Risk (Critical Intervention)", text: "text-rose-300", stroke: "stroke-rose-400", bg: "bg-rose-400" };

/* ------------------------------------------------------------------
   UI Components (Clean, Professional, No Emojis)
------------------------------------------------------------------ */
const Panel = ({ title, note, children, className = "" }) => (
  <section className={`rounded-lg border border-slate-800 bg-slate-900 p-5 ${className}`}>
    {title && <h2 className="text-base font-semibold tracking-tight text-slate-100">{title}</h2>}
    {note && <p className="mt-1 text-xs text-slate-400">{note}</p>}
    <div className={title ? "mt-4" : ""}>{children}</div>
  </section>
);

function Field({ f, raw, onChange, disabled }) {
  const err = errorFor(f, raw);
  const sliderVal = err ? f.min : Number(raw);
  const id = `f-${f.key}`;
  return (
    <div className={disabled ? "opacity-40 pointer-events-none" : ""}>
      <div className="flex items-center justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium text-slate-200">{f.label}</label>
        <div className="flex items-center gap-2">
          <input
            id={id} inputMode="decimal" value={raw}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            aria-invalid={!!err} aria-describedby={`${id}-msg`}
            className={`w-20 rounded-md border bg-slate-950 px-2 py-1.5 text-right tabular-nums text-slate-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-teal-400 ${err ? "border-rose-400" : "border-slate-700"}`}
          />
          <span className="w-24 text-xs text-slate-400">{f.unit}</span>
        </div>
      </div>
      <input
        type="range" min={f.min} max={f.max} step={f.step} value={sliderVal}
        onChange={(e) => onChange(e.target.value)} aria-label={`${f.label} slider`}
        disabled={disabled}
        className="mt-2 w-full accent-teal-400"
      />
      <p id={`${id}-msg`} role={err ? "alert" : undefined} className={`mt-1 text-xs ${err ? "text-rose-300" : "text-slate-500"}`}>
        {disabled ? "Evaluated in Weeks 7+ after midterms" : err || f.hint}
      </p>
    </div>
  );
}

function ScoreRing({ score }) {
  const r = 68, c = 2 * Math.PI * r, b = band(score);
  return (
    <div className="relative h-44 w-44 shrink-0">
      <svg viewBox="0 0 160 160" className="h-full w-full -rotate-90" role="img" aria-label={`Predicted score ${score.toFixed(1)} out of 100`}>
        <circle cx="80" cy="80" r={r} fill="none" strokeWidth="10" className="stroke-slate-800" />
        <circle
          cx="80" cy="80" r={r} fill="none" strokeWidth="10" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - score / 100)}
          className={`${b.stroke} transition-[stroke-dashoffset] duration-500 motion-reduce:transition-none`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-semibold tabular-nums text-slate-100">{score.toFixed(1)}</span>
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">out of 100</span>
      </div>
    </div>
  );
}

function RangeBar({ score, mae }) {
  const lo = clamp(score - mae), hi = clamp(score + mae), b = band(score);
  return (
    <div className="mt-4" aria-hidden="true">
      <div className="relative h-2 rounded-full bg-slate-800">
        <div className={`absolute h-2 rounded-full opacity-40 ${b.bg}`} style={{ left: `${lo}%`, width: `${hi - lo}%` }} />
        <div className={`absolute -top-1 h-4 w-1 rounded-full ${b.bg}`} style={{ left: `calc(${score}% - 2px)` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-500"><span>0 (Floor)</span><span>70 (Passing)</span><span>100</span></div>
    </div>
  );
}

function Contributions({ parts }) {
  const max = Math.max(...parts.map((p) => Math.abs(p.delta)), 1);
  return (
    <ul className="space-y-3">
      {parts.map((p) => (
        <li key={p.key} className="grid grid-cols-[10rem_1fr_3.5rem] items-center gap-3 text-xs">
          <span className="text-slate-300">{p.label}</span>
          <div className="relative h-2 rounded-full bg-slate-800">
            <div className="absolute left-1/2 top-[-3px] h-[14px] w-px bg-slate-600" />
            <div
              className={`absolute h-2 rounded-full transition-all duration-300 motion-reduce:transition-none ${p.delta >= 0 ? "bg-teal-400" : "bg-rose-400"}`}
              style={p.delta >= 0 ? { left: "50%", width: `${(p.delta / max) * 50}%` } : { right: "50%", width: `${(-p.delta / max) * 50}%` }}
            />
          </div>
          <span className="text-right tabular-nums font-mono text-slate-300">{p.delta >= 0 ? "+" : ""}{p.delta.toFixed(1)}</span>
        </li>
      ))}
    </ul>
  );
}

/* ------------------------------------------------------------------
   Views
------------------------------------------------------------------ */
function PredictView() {
  const [isEarly, setIsEarly] = useState(false);
  const [raw, setRaw] = useState(PRESETS["Average student"]);
  const [preset, setPreset] = useState("Average student");

  const set = (k) => (val) => { setPreset(null); setRaw((r) => ({ ...r, [k]: val })); };
  
  const activeFields = isEarly ? ALL_FIELDS.filter((f) => !f.isMidterm) : ALL_FIELDS;
  const valid = activeFields.every((f) => !errorFor(f, raw[f.key]));
  const nums = useMemo(() => Object.fromEntries(ALL_FIELDS.map((f) => [f.key, Number(raw[f.key])])), [raw]);
  
  const result = useMemo(() => (valid ? predict(nums, isEarly) : null), [valid, nums, isEarly]);
  const tips = useMemo(() => (valid ? advice(nums, isEarly) : []), [valid, nums, isEarly]);
  const anomaly = useMemo(() => (!isEarly && valid ? checkAnomaly(nums) : { isAnomaly: false }), [isEarly, valid, nums]);

  const activeMae = isEarly ? EARLY_MAE : FULL_MAE;

  return (
    <div className="space-y-6">
      {/* Milestone Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-800 bg-slate-900 px-4 py-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Semester Milestone Mode</p>
          <p className="text-xs text-slate-500">Switch between pre-midterm intervention (Weeks 1-6) and full evaluation (Weeks 7+).</p>
        </div>
        <div className="inline-flex rounded-md border border-slate-700 p-0.5" role="group">
          <button
            type="button" onClick={() => setIsEarly(true)}
            className={`rounded px-3 py-1.5 text-xs font-medium focus-visible:outline-none ${isEarly ? "bg-slate-800 text-teal-300 font-semibold" : "text-slate-400 hover:text-slate-200"}`}
          >
            Pre-Midterm (Weeks 1-6)
          </button>
          <button
            type="button" onClick={() => setIsEarly(false)}
            className={`rounded px-3 py-1.5 text-xs font-medium focus-visible:outline-none ${!isEarly ? "bg-slate-800 text-teal-300 font-semibold" : "text-slate-400 hover:text-slate-200"}`}
          >
            Post-Midterm (Weeks 7+)
          </button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,26rem)_1fr]">
        <Panel title="Student Indicators" note={isEarly ? "Pre-midterm metrics. Midterm input deactivated." : "Comprehensive pre-final metrics."}>
          <div className="mb-5 flex flex-wrap gap-2" role="group" aria-label="Preset profiles">
            {Object.keys(PRESETS).map((name) => (
              <button
                key={name} type="button" aria-pressed={preset === name}
                onClick={() => { setRaw(PRESETS[name]); setPreset(name); }}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium focus-visible:outline-none ${preset === name ? "border-teal-400 text-teal-300 bg-slate-800" : "border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200"}`}
              >{name}</button>
            ))}
          </div>

          <div className="space-y-4">
            {ALL_FIELDS.map((f) => (
              <Field
                key={f.key} f={f} raw={raw[f.key]}
                onChange={set(f.key)}
                disabled={isEarly && f.isMidterm}
              />
            ))}
          </div>
          <button
            type="button" onClick={() => { setRaw(PRESETS["Average student"]); setPreset("Average student"); }}
            className="mt-6 text-xs text-slate-400 underline underline-offset-4 hover:text-slate-200 focus-visible:outline-none"
          >Reset to cohort average</button>
        </Panel>

        <div className="space-y-6">
          {/* Anomaly Detection Banner */}
          {anomaly.isAnomaly && (
            <div className="rounded-lg border border-amber-500/40 bg-amber-950/20 p-4 text-xs text-amber-200">
              <p className="font-semibold uppercase tracking-wider text-amber-400">Midterm Exam Anomaly Detected</p>
              <p className="mt-1 leading-relaxed text-amber-200/90">
                Coursework average ({nums.assignment}%) and GPA ({nums.gpa}) projected an expected midterm of ~{anomaly.expected} points, but actual midterm was {nums.midterm} (-{anomaly.deficit} pt deficit). This divergence indicates acute situational factors (such as illness or exam anxiety) rather than baseline inability. Consider counselor follow-up.
              </p>
            </div>
          )}

          <Panel>
            <div aria-live="polite">
              {result ? (
                <div className="flex flex-col items-center gap-6 sm:flex-row">
                  <ScoreRing score={result.score} />
                  <div className="w-full">
                    <div className="flex items-center gap-2">
                      <span className={`inline-block h-2.5 w-2.5 rounded-full ${band(result.score).bg}`} />
                      <p className={`text-base font-semibold ${band(result.score).text}`}>{band(result.score).name}</p>
                    </div>
                    <p className="mt-1 text-sm text-slate-300">
                      Expected score interval: <span className="font-mono font-semibold text-slate-100">{clamp(result.score - activeMae).toFixed(1)}</span> to <span className="font-mono font-semibold text-slate-100">{clamp(result.score + activeMae).toFixed(1)}</span>
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      {isEarly ? `Pre-Midterm early model (MAE: ${activeMae} pts, R2: 0.916)` : `Full model (MAE: ${activeMae} pts, R2: 0.929)`}
                    </p>
                    <RangeBar score={result.score} mae={activeMae} />
                  </div>
                </div>
              ) : (
                <p className="py-10 text-center text-xs text-slate-400">Correct invalid inputs to calculate prediction.</p>
              )}
            </div>
          </Panel>

          {result && (
            <>
              <Panel title="Feature Marginal Impact" note="Points added or subtracted relative to cohort mean performance.">
                <Contributions parts={result.parts} />
              </Panel>
              <Panel title="Actionable Guidance & Interventions" note="Prescriptive academic recovery guidance based on feature sensitivities.">
                <ul className="space-y-3">
                  {tips.map(([h, t]) => (
                    <li key={h} className="border-l-2 border-slate-700 pl-3">
                      <p className="text-sm font-medium text-slate-100">{h}</p>
                      <p className="mt-0.5 text-xs leading-relaxed text-slate-400">{t}</p>
                    </li>
                  ))}
                </ul>
              </Panel>
            </>
          )}
          <p className="text-xs text-slate-500">Estimates are statistical associations derived from historical training data and do not determine formal institutional grades.</p>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------
   Model Comparison View
------------------------------------------------------------------ */
const REAL_MODELS = [
  { name: "Post-Midterm Multiple Linear Regression (Champion)", mae: "2.63", rmse: "3.39", r2: "0.929", cv: "0.882 (0.018)", status: "Full Model" },
  { name: "Pre-Midterm Early Stage Model (Weeks 1-6)", mae: "3.00", rmse: "3.69", r2: "0.916", cv: "0.865 (0.021)", status: "Early Intervention" },
  { name: "Post-Midterm Random Forest Regressor", mae: "3.15", rmse: "4.02", r2: "0.900", cv: "0.854 (0.026)", status: "Comparison" },
  { name: "Post-Midterm Ridge Regression (alpha=1.0)", mae: "2.63", rmse: "3.39", r2: "0.929", cv: "0.882 (0.018)", status: "Regularized" },
];

const REAL_IMPORTANCE = [
  { label: "Midterm exam score", lr: 0.406, rf: 0.441 },
  { label: "Dedicated study hours", lr: 0.219, rf: 0.098 },
  { label: "Prior cumulative GPA", lr: 0.206, rf: 0.119 },
  { label: "Assignment average", lr: 0.203, rf: 0.236 },
  { label: "Classroom attendance", lr: 0.198, rf: 0.107 },
];

const TEST_SCATTER_POINTS = [
  [90.1, 87.6], [75.7, 76.6], [94.0, 92.7], [89.2, 89.0], [58.8, 60.8],
  [86.3, 85.4], [70.8, 70.1], [79.0, 79.2], [66.3, 72.1], [83.0, 77.3],
  [93.7, 88.4], [79.8, 77.8], [64.2, 66.7], [90.6, 95.5], [64.1, 65.9],
  [75.0, 78.5], [96.0, 96.4], [75.1, 72.9], [80.1, 84.5], [69.6, 68.0],
  [50.5, 53.9], [68.7, 64.2], [95.1, 93.6], [61.4, 64.6], [82.9, 81.2],
  [54.0, 57.8], [74.0, 80.2], [58.9, 54.8], [62.2, 65.3], [97.4, 98.4],
  [69.8, 76.8], [83.4, 80.5], [63.8, 65.9], [82.2, 82.7], [89.3, 91.0],
  [99.3, 93.6], [79.8, 80.9], [81.9, 79.8], [67.1, 67.0], [86.6, 87.3],
  [81.3, 77.4], [71.3, 71.8], [80.7, 82.4], [65.4, 63.8], [66.8, 67.9]
];

function InsightsView() {
  const [modelType, setModelType] = useState("lr");

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Feature Driver Comparison" note="Parametric Standardized Betas vs. Non-linear Tree MDI Gini Importances.">
          <div className="mb-4 inline-flex rounded-md border border-slate-700 p-0.5" role="group" aria-label="Model selector">
            {[["lr", "Multiple Linear Regression"], ["rf", "Random Forest"]].map(([k, n]) => (
              <button
                key={k} type="button" aria-pressed={modelType === k} onClick={() => setModelType(k)}
                className={`rounded px-3 py-1 text-xs font-medium focus-visible:outline-none ${modelType === k ? "bg-slate-800 text-slate-100" : "text-slate-400 hover:text-slate-200"}`}
              >{n}</button>
            ))}
          </div>
          <ul className="space-y-3">
            {REAL_IMPORTANCE.map((r) => (
              <li key={r.label} className="grid grid-cols-[10rem_1fr_3.5rem] items-center gap-3 text-xs">
                <span className="text-slate-300">{r.label}</span>
                <div className="h-2 rounded-full bg-slate-800">
                  <div
                    className="h-2 rounded-full bg-teal-400 transition-all duration-300 motion-reduce:transition-none"
                    style={{ width: `${(r[modelType] / 0.45) * 100}%` }}
                  />
                </div>
                <span className="text-right font-mono tabular-nums text-slate-300">{r[modelType].toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Actual vs. Predicted Performance" note="Holdout test cohort (80 students). Closeness to dashed 45-degree line indicates accuracy.">
          <svg viewBox="0 0 240 240" className="mx-auto w-full max-w-xs" role="img" aria-label="Scatter of actual versus predicted scores">
            <rect x="30" y="10" width="200" height="200" className="fill-none stroke-slate-800" />
            <line x1="30" y1="210" x2="230" y2="10" className="stroke-slate-600" strokeDasharray="3 3" />
            {TEST_SCATTER_POINTS.map(([a, p], i) => (
              <circle
                key={i}
                cx={30 + ((a - 30) / 70) * 200}
                cy={210 - ((p - 30) / 70) * 200}
                r="3"
                className="fill-teal-400 opacity-80"
              />
            ))}
            <text x="130" y="232" textAnchor="middle" className="fill-slate-400 text-[10px]">Actual Holdout Score</text>
            <text transform="translate(14 110) rotate(-90)" textAnchor="middle" className="fill-slate-400 text-[10px]">Predicted Score</text>
          </svg>
          <div className="mt-2 text-center text-xs text-slate-500">Holdout RMSE: 3.39 points | R2: 0.929</div>
        </Panel>
      </div>

      <Panel title="Benchmark Model Comparison Across Milestones" note="Evaluated on 80 holdout testing records with 5-fold cross-validation on train split.">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[34rem] text-left text-xs">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="py-2 pr-4 font-medium">Model Architecture</th>
                <th className="py-2 pr-4 text-right font-medium">Test MAE</th>
                <th className="py-2 pr-4 text-right font-medium">Test RMSE</th>
                <th className="py-2 pr-4 text-right font-medium">Test R2</th>
                <th className="py-2 pr-4 text-right font-medium">5-Fold CV R2</th>
                <th className="py-2 text-right font-medium">Milestone Role</th>
              </tr>
            </thead>
            <tbody>
              {REAL_MODELS.map((m) => (
                <tr key={m.name} className="border-t border-slate-800/60">
                  <td className="py-2.5 pr-4 font-medium text-slate-200">{m.name}</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-300">{m.mae}</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-300">{m.rmse}</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-300">{m.r2}</td>
                  <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-slate-300">{m.cv}</td>
                  <td className="py-2.5 text-right font-mono text-teal-300">{m.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function AboutView() {
  const items = [
    ["Dual-Stage Architecture", "Provides two specialized models: an Early-Stage Model (Weeks 1-6, R2=0.916) for catching at-risk students before midterms, and a Post-Midterm Model (Weeks 7+, R2=0.929) for fine-grained prediction."],
    ["Exam Anomaly Detection", "Detects situational test anxiety or acute illness when coursework trajectory deviates significantly from midterm exam marks, safeguarding students against single-exam penalization."],
    ["Anti-Leakage Guarantees", "80/20 train/test partition executed prior to scaling. Standard scalers fit strictly on training samples to eliminate test set contamination."],
    ["Ethical Use in Advising", "All estimates are statistical associations designed as advisory indicators to empower student intervention rather than automated pass/fail determinations."],
  ];
  return (
    <div className="max-w-2xl space-y-4">
      {items.map(([h, t]) => (
        <Panel key={h} title={h}><p className="text-xs leading-relaxed text-slate-300">{t}</p></Panel>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------
   Shell & Top Navigation
------------------------------------------------------------------ */
const NAV = [
  { id: "predict", label: "Predictor", title: "Student Score Predictor", sub: "Interactive student profile simulator, milestone toggle, and anomaly triage." },
  { id: "insights", label: "Model Insights", title: "Model Performance & Drivers", sub: "Holdout benchmark metrics across pre- and post-midterm milestones." },
  { id: "about", label: "Documentation", title: "System Architecture & Limits", sub: "Dual-stage formulation, anomaly logic, and ethical governance." },
];

export default function App() {
  const [view, setView] = useState("predict");
  const cur = NAV.find((n) => n.id === view);

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-200 md:flex">
      <aside className="border-b border-slate-800 p-4 md:min-h-screen md:w-60 md:shrink-0 md:border-b-0 md:border-r md:p-6">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-teal-400" />
          <p className="font-semibold tracking-tight text-slate-100">Score Predictor</p>
        </div>
        <p className="mb-6 mt-1 text-xs text-slate-400">Academic Early Warning System</p>
        <nav aria-label="Main" className="flex gap-1 md:flex-col">
          {NAV.map((n) => (
            <button
              key={n.id} type="button" onClick={() => setView(n.id)} aria-current={view === n.id ? "page" : undefined}
              className={`rounded-md px-3 py-2 text-left text-xs font-medium focus-visible:outline-none ${view === n.id ? "bg-slate-800 text-teal-300 font-semibold" : "text-slate-400 hover:text-slate-200"}`}
            >
              {n.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="min-w-0 flex-1 p-4 md:p-8">
        <header className="mb-6 border-b border-slate-800 pb-4">
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">{cur.title}</h1>
          <p className="mt-1 text-xs text-slate-400">{cur.sub}</p>
        </header>
        {view === "predict" && <PredictView />}
        {view === "insights" && <InsightsView />}
        {view === "about" && <AboutView />}
      </main>
    </div>
  );
}
