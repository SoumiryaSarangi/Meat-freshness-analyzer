# Experimental Validation Plan & Protocols

**Project:** Marbl — AI Meat Quality Control & Predictive Shelf-Life Inspection Pipeline  
**Document:** Industrial Laboratory & Production Plant Experimental Verification Protocols  

---

## 1. Validation Strategy Overview

To transition Marbl from software validation to production facility acceptance, a four-protocol experimental verification plan is established. Each protocol benchmarks an AI module against certified laboratory gold standards (ISO, AOAC, and calibrated metrology).

```mermaid
flowchart TD
    subgraph P1 ["Protocol 1: Freshness Classification"]
        A1["200 Industrial Meat Samples<br/>(Beef, Pork, Poultry)"] --> B1["Dual Blind Inspection:<br/>ISO 4833-1 Plate Counts vs.<br/>MobileNetV2 Inference"]
        B1 --> C1["Success Gate:<br/>Accuracy ≥ 94%<br/>Spoilage Recall ≥ 96%"]
    end

    subgraph P2 ["Protocol 2: Dimensional Metrology"]
        A2["100 Primal Meat Cuts<br/>Varying Distance & Tilt"] --> B2["Caliper Verification:<br/>Mitutoyo Vernier (±0.02mm)<br/>vs. ISO ID-1 Vision Sizing"]
        B2 --> C2["Success Gate:<br/>MAPE ≤ 3.0%<br/>Routing Accuracy ≥ 97%"]
    end

    subgraph P3 ["Protocol 3: Shelf-Life Kinetics"]
        A3["Environmental Incubator Trials<br/>(0°C, 4°C, 8°C, 12°C)"] --> B3["Serial Microbial Sampling:<br/>Daily TVC Enumeration vs.<br/>Ratkowsky Kinetic Curve"]
        B3 --> C3["Success Gate:<br/>R² ≥ 0.90<br/>Absolute Error ≤ 1.5 Days"]
    end

    subgraph P4 ["Protocol 4: Lipid Extraction"]
        A4["50 Beef & Pork Steaks<br/>Varying Marbling Levels"] --> B4["Chemical Analysis:<br/>AOAC 991.36 Soxhlet Ether<br/>Extraction vs. Visual Fat %"]
        B4 --> C4["Success Gate:<br/>Pearson r ≥ 0.88<br/>p < 0.001"]
    end
```

---

## 2. Experimental Protocols

### Protocol 1: Freshness Classification Validation (ISO 4833-1 Comparison)
- **Objective:** Statistically validate MobileNetV2 binary freshness classification against certified microbiological plate counts.
- **Sample Cohort:** $N = 200$ fresh and aged cuts (beef strip loin, chicken breast, pork chop, ground meat) sampled at loading docks.
- **Reference Standard:** **ISO 4833-1:2013** (*Microbiology of the food chain — Horizontal method for the enumeration of microorganisms — Colony count at 30°C*).
- **Execution:**
  1. High-resolution photograph acquired under standardized plant lighting.
  2. A $25\text{ g}$ aseptic surface excision taken from the imaged cut, homogenized with $225\text{ mL}$ buffered peptone water.
  3. Serial dilutions plated on Plate Count Agar (PCA) and incubated at $30^\circ\text{C}$ for $72\text{ hours}$.
  4. Spoilage threshold defined at $\log_{10}(\text{CFU/g}) \ge 7.0$.
- **Acceptance Criteria:** Binary classification accuracy $\ge 94\%$; recall on spoiled samples $\ge 96\%$.

### Protocol 2: Dimensional Metrology Validation (Vernier Caliper Comparison)
- **Objective:** Quantify measurement error of ISO ID-1 optical calibration under industrial viewing conditions.
- **Sample Cohort:** $N = 100$ primal cuts and portions across varied thicknesses ($15\text{ mm}$ to $80\text{ mm}$) and lengths ($40\text{ mm}$ to $350\text{ mm}$).
- **Reference Standard:** Calibrated **Mitutoyo 500-196-30 Digital Vernier Caliper** ($\pm 0.02\text{ mm}$ accuracy).
- **Execution:**
  1. Longest dimension measured physically with calipers by two independent metrologists.
  2. Samples placed on conveyor / cutting board alongside an ISO/IEC 7810 ID-1 card.
  3. Images captured across test angles ($0^\circ$, $15^\circ$, $30^\circ$) and heights ($30\text{ cm}$ to $70\text{ cm}$).
  4. Error calculated as $\Delta L = |L_{\text{vision}} - L_{\text{caliper}}|$.
- **Acceptance Criteria:** $\text{MAPE} \le 3.0\%$; routing classification agreement $\ge 97\%$.

### Protocol 3: Kinetic Shelf-Life Environmental Incubation Validation
- **Objective:** Validate the Ratkowsky square-root kinetic engine under isothermal and dynamic cold-chain regimes.
- **Sample Cohort:** 40 vacuum-packaged and 40 air-permeable packaged beef loin cuts from identical slaughter batches ($pH_u = 5.60 \pm 0.05$).
- **Reference Standard:** Daily Total Viable Count (TVC) enumeration in certified environmental test chambers.
- **Execution:**
  1. Batches distributed across constant-temperature incubators: $0^\circ\text{C}$, $4^\circ\text{C}$, $8^\circ\text{C}$, and $12^\circ\text{C}$.
  2. Triplicate destructive microbial plating performed every 24 hours until TVC exceeds $10^7\text{ CFU/g}$.
  3. Predicted shelf-life curve $t_{\text{shelf-life}}(T, pH, \text{pkg})$ compared against observed empirical time-to-spoilage.
- **Acceptance Criteria:** Coefficient of determination $R^2 \ge 0.90$; prediction deviation $\le \pm 1.5\text{ days}$.

### Protocol 4: Visible Fat Coverage vs. Chemical Soxhlet Extraction
- **Objective:** Correlate optical visible fat coverage percentage against total chemical lipid content.
- **Sample Cohort:** $N = 50$ beef ribeye and pork loin steaks displaying fat coverage from $5\%$ to $45\%$.
- **Reference Standard:** **AOAC Official Method 991.36** (*Fat [Crude] in Meat and Meat Products — Solvent Extraction*).
- **Execution:**
  1. High-resolution top-down photograph analyzed by Marbl Module 3 colorimetric algorithm.
  2. Entire surface trimmed and homogenized via industrial food processor.
  3. Solvent extraction performed using petroleum ether in a Soxhlet apparatus for 4 hours; dried to constant weight.
  4. Pearson correlation coefficient ($r$) computed between visible fat area $\%$ and chemical crude lipid $\%$.
- **Acceptance Criteria:** Pearson correlation $r \ge 0.88$ ($p < 0.001$).

