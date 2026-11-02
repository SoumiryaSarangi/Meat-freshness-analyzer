// shared.js -- pure logic, no DOM references

// ── Client-side predictive shelf life engine (instant 0ms offline capability) ──
const SPECIES_PROFILE = {
  beef: { mu_ref_day: 1.45, t_min: -3.5, alpha: 0.85, max_vac_days: 42.0, max_air_days: 10.0, name: "Beef" },
  pork: { mu_ref_day: 1.70, t_min: -3.0, alpha: 1.00, max_vac_days: 28.0, max_air_days: 8.0, name: "Pork" },
  mutton: { mu_ref_day: 1.55, t_min: -3.2, alpha: 0.92, max_vac_days: 32.0, max_air_days: 9.0, name: "Mutton" },
  chicken: { mu_ref_day: 2.25, t_min: -2.5, alpha: 1.45, max_vac_days: 16.0, max_air_days: 6.0, name: "Chicken" }
};

const PACKAGING_PROFILE = {
  air: { beta: 1.00, log_spoil: 7.0, desc: "Air (Aerobic)" },
  map: { beta: 0.65, log_spoil: 7.2, desc: "Modified Atmosphere Packaging (MAP)" },
  vacuum: { beta: 0.45, log_spoil: 7.6, desc: "Vacuum Packaging (VP)" }
};

function calcRatkowskyMu(tC, tMin, muRef, refT = 4.0) {
  if (tC <= tMin) return 0.0;
  const val = (tC - tMin) / (refT - tMin);
  let rate = muRef * (val * val);
  if (tC > 10.0) rate *= (1.0 + 0.05 * (tC - 10.0));
  return Math.max(0.001, rate);
}

function calcPhFactor(ph) {
  if (ph >= 5.6) {
    return 1.0 + 1.8 * (ph - 5.6);
  } else {
    return Math.max(0.15, 1.0 - 1.5 * (5.6 - ph));
  }
}

export function calculateLocalShelfLife(data) {
  const spKey = (data.meat_type || 'beef').toLowerCase().trim();
  const pkgKey = (data.packaging_type || 'air').toLowerCase().trim();
  const sp = SPECIES_PROFILE[spKey] || SPECIES_PROFILE.beef;
  const pkg = PACKAGING_PROFILE[pkgKey] || PACKAGING_PROFILE.air;

  const ph = Math.max(4.8, Math.min(7.2, Number(data.initial_ph) || 5.62));
  const recTemp = Number(data.receiving_temp_c) ?? 2.0;
  const timeSlaughter = Math.max(1.0, Number(data.time_since_slaughter_hrs) || 24.0);
  const transDur = Math.max(0.5, Number(data.transport_duration_hrs) || 4.0);
  let transTemp = Number(data.transport_temp_avg_c) ?? 2.0;
  if (data.transport_excursion) transTemp += 3.5;

  const muRef = sp.mu_ref_day;
  const tMin = sp.t_min;
  const alphaSpecies = sp.alpha;
  const betaPkg = pkg.beta;
  const gammaPh = calcPhFactor(ph);

  const logN0 = 2.8;
  const chillerHoldingHrs = Math.max(0.0, timeSlaughter - transDur);
  const muChiller = calcRatkowskyMu(2.0, tMin, muRef) * alphaSpecies * betaPkg * gammaPh;
  const deltaLogChiller = (muChiller / 24.0 * chillerHoldingHrs) / Math.LN10;

  const muTrans = calcRatkowskyMu(transTemp, tMin, muRef) * alphaSpecies * betaPkg * gammaPh;
  const deltaLogTrans = (muTrans / 24.0 * transDur) / Math.LN10;

  const currentLogCfu = logN0 + deltaLogChiller + deltaLogTrans;
  const effectiveCurrentLog = Math.min(pkg.log_spoil - 0.1, currentLogCfu);

  const recHoldingTemp = 1.0;
  const muHolding = calcRatkowskyMu(recHoldingTemp, tMin, muRef) * alphaSpecies * betaPkg * gammaPh;

  const logHeadroom = Math.max(0.1, pkg.log_spoil - effectiveCurrentLog);
  const lnHeadroom = logHeadroom * Math.LN10;
  const rawDays = lnHeadroom / muHolding;

  let ceiling = sp.max_air_days;
  if (pkgKey === 'vacuum') ceiling = sp.max_vac_days;
  else if (pkgKey === 'map') ceiling = sp.max_vac_days * 0.58;

  let remainingDays = Math.min(ceiling, rawDays);

  if (recTemp > 4.0) {
    const cooldownPenalty = (recTemp - 4.0) * 0.25;
    remainingDays = Math.max(0.3, remainingDays - cooldownPenalty);
  }

  const varianceFactor = 0.16;
  const minDays = Math.max(0.2, remainingDays * (1.0 - varianceFactor));
  const maxDays = remainingDays * (1.0 + varianceFactor);

  let conf = 0.90;
  if (data.transport_excursion || recTemp > 6.0) conf -= 0.15;
  if (ph > 6.2) conf -= 0.08;
  conf = Math.max(0.55, Math.min(0.96, conf));

  const warnings = [];
  if (recTemp > 4.0) {
    warnings.push(`🚨 Cold Chain Breach: Receiving temperature is ${recTemp.toFixed(1)}°C (> 4.0°C CCP threshold). Rapid chilling required.`);
  } else if (recTemp > 2.5) {
    warnings.push(`⚠️ Elevated Receiving Temp: Meat arrived at ${recTemp.toFixed(1)}°C. Ideal receiving temperature is 0°C to +2°C.`);
  }

  if (data.transport_excursion) {
    warnings.push('🚨 Transport Excursion: Temperature break logged during shipping. Microbial proliferation accelerated.');
  } else if (transTemp > 4.5) {
    warnings.push(`⚠️ High Transit Temp: Average transport was ${transTemp.toFixed(1)}°C (> 4.0°C standard).`);
  }

  if (ph >= 6.0) {
    warnings.push(`⚠️ High pH / DFD Spoilage Risk: Initial pH ${ph.toFixed(2)} (DFD meat). Reduced lactic acid defense accelerates microbial spoilage.`);
  } else if (ph < 5.4 && spKey === 'pork') {
    warnings.push(`ℹ️ Low pH (PSE Susceptibility): Initial pH ${ph.toFixed(2)}. Potential pale/soft exudative meat with higher drip tendency.`);
  }

  if (remainingDays < 2.0) {
    warnings.push(`🚨 Critical Shelf Life: Less than 48 hours remaining (${remainingDays.toFixed(1)} days). Prioritize immediate processing.`);
  }

  let recCond = "Store at 0°C to +2°C with high relative humidity (85-90%).";
  if (pkgKey === 'vacuum') recCond = "Store at 0°C to +2°C in dark chill room. Avoid packaging puncture; monitor purge.";
  else if (pkgKey === 'map') recCond = "Store at 0°C to +2°C under continuous chilling. Prevent gas seal damage.";

  return {
    shelf_life_days: Number(remainingDays.toFixed(1)),
    shelf_life_min_days: Number(minDays.toFixed(1)),
    shelf_life_max_days: Number(maxDays.toFixed(1)),
    confidence_score: Number(conf.toFixed(2)),
    recommended_temp_c: "0°C to +2°C",
    recommended_conditions: recCond,
    warnings: warnings
  };
}

export async function estimateShelfLife(intakeData) {
  try {
    const res = await fetch('/api/shelf-life/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(intakeData)
    });
    if (res.ok) return await res.json();
  } catch (e) {
    // Fall back to local calculation if offline or server loading
  }
  return calculateLocalShelfLife(intakeData);
}

export async function classifyImages(files, batchId = null) {
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));
  if (batchId) {
    formData.append('batch_id', batchId);
  }
  const res = await fetch('/api/classify', { method: 'POST', body: formData });
  if (!res.ok) throw new Error(`Server responded ${res.status}`);
  return (await res.json()).results;
}

export async function createBatch(batchData) {
  try {
    const res = await fetch('/api/batch/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(batchData)
    });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn('Backend server offline during batch create; saving locally:', err);
  }

  // Graceful local fallback if backend is offline
  const localShelf = calculateLocalShelfLife(batchData);
  const localBatch = {
    ...batchData,
    created_at: new Date().toISOString(),
    shelf_life_days: localShelf.shelf_life_days,
    shelf_life_min_days: localShelf.shelf_life_min_days,
    shelf_life_max_days: localShelf.shelf_life_max_days,
    confidence_score: localShelf.confidence_score,
    recommended_temp_c: localShelf.recommended_temp_c,
    warnings: localShelf.warnings,
    status: 'intake_complete (local)'
  };
  return { batch: localBatch, shelf_life: localShelf };
}

export async function getBatch(batchId) {
  const res = await fetch(`/api/batch/${encodeURIComponent(batchId)}`);
  if (!res.ok) throw new Error(`Failed to load batch: ${res.status}`);
  return (await res.json()).batch;
}

export async function listBatches() {
  try {
    const res = await fetch('/api/batches');
    if (!res.ok) return [];
    return (await res.json()).batches;
  } catch (e) {
    return [];
  }
}

export async function getInspectionRecords(batchId) {
  try {
    const res = await fetch(`/api/batch/${encodeURIComponent(batchId)}/records`);
    if (!res.ok) return [];
    return (await res.json()).records;
  } catch (e) {
    return [];
  }
}

export function downloadAnnotatedImage(base64DataUrl, filename) {
  const b64 = base64DataUrl.replace('data:image/jpeg;base64,', '');
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  const blob = new Blob([arr], { type: 'image/jpeg' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `annotated_${filename.replace(/\.[^.]+$/, '')}.jpg`;
  a.click();
  URL.revokeObjectURL(a.href);
}
