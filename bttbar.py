"""
cut_analysis_WWjj.py
====================
Aplica os cortes cinemáticos sequenciais do artigo (Eq. 5) ao arquivo ROOT
gerado pelo Delphes para o processo pp > jj W+ W+ (W+ > l+ vl) ou backgrounds (ttbar).

Cortes aplicados (SS leptons):
  (I)   Pelo menos 1 b-jet com pT > 20 GeV e |eta| < 2.5
  (II)  MET > 20 GeV
  (III) Exatamente 2 léptons SS com pT > 10 GeV e |eta| < 2.5
  (IV)  Massa invariante dilépton M_ll > 100 GeV
"""

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import uproot
    import awkward as ak
    import vector
    vector.register_awkward()  # Permite operações de quadrivetores direto no Awkward
    USE_UPROOT = True
except ImportError:
    USE_UPROOT = False

try:
    import ROOT
    USE_ROOT = True
except ImportError:
    USE_ROOT = False

if not USE_UPROOT and not USE_ROOT:
    raise ImportError(
        "Instale uproot+awkward+vector ou ROOT/PyROOT.\n"
        "  pip install uproot awkward vector numpy"
    )


def analyze_uproot(input_file: str, tree_name: str = "Delphes"):
    print(f"\n[uproot] Lendo: {input_file}  (tree: {tree_name})")
    f = uproot.open(input_file)
    tree = f[tree_name]

    # ---- Leitura dos branches necessários ----
    branches = [
        "Jet.PT", "Jet.Eta", "Jet.BTag",
        "MissingET.MET",
        "Electron.PT", "Electron.Eta", "Electron.Charge", "Electron.Phi",
        "Muon.PT", "Muon.Eta", "Muon.Charge", "Muon.Phi",
    ]

    data = tree.arrays(branches, library="ak")
    N_total = len(data)
    print(f"  Eventos totais: {N_total}")

    # =========================================================
    # CORTE (I): >= 1 b-jet com pT > 20 GeV e |eta| < 2.5
    # =========================================================
    jet_pt   = data["Jet.PT"]
    jet_eta  = data["Jet.Eta"]
    jet_btag = data["Jet.BTag"]

    good_bjet = (jet_pt > 20.0) & (np.abs(jet_eta) < 2.5) & (jet_btag == 1)
    n_bjets   = ak.sum(good_bjet, axis=1)
    mask_I    = n_bjets >= 1

    N_I = ak.sum(mask_I)
    eff_I = float(N_I) / N_total
    print(f"\n  [Corte I]  >= 1 b-jet (pT>20, |eta|<2.5)")
    print(f"             Eventos: {N_I}  |  Eficiência: {eff_I:.4f}")

    # =========================================================
    # CORTE (II): MET > 20 GeV
    # =========================================================
    met      = data["MissingET.MET"]
    met_val  = ak.firsts(met, axis=1)
    met_val  = ak.fill_none(met_val, 0.0)

    mask_II  = mask_I & (met_val > 20.0)
    N_II     = ak.sum(mask_II)
    eff_II   = float(N_II) / N_total
    print(f"\n  [Corte II] MET > 20 GeV")
    print(f"             Eventos: {N_II}  |  Eficiência: {eff_II:.4f}")

    # =========================================================
    # CORTE (III): Exatamente 2 léptons SS com pT > 10 GeV, |eta| < 2.5
    # =========================================================
    # CORREÇÃO: Indentação corrigida aqui para dentro da função
    good_el = (data["Electron.PT"] > 10.0) & (np.abs(data["Electron.Eta"]) < 2.5)
    good_mu = (data["Muon.PT"]     > 10.0) & (np.abs(data["Muon.Eta"])     < 2.5)

    # Monta quadrivetores usando a nomenclatura do pacote 'vector' (com momentum4D)
    el_vec = ak.zip({
        "pt":     data["Electron.PT"][good_el],
        "eta":    data["Electron.Eta"][good_el],
        "phi":    data["Electron.Phi"][good_el],
        "mass":   ak.zeros_like(data["Electron.PT"][good_el]),
        "charge": data["Electron.Charge"][good_el],
    }, with_name="Momentum4D")

    mu_vec = ak.zip({
        "pt":     data["Muon.PT"][good_mu],
        "eta":    data["Muon.Eta"][good_mu],
        "phi":    data["Muon.Phi"][good_mu],
        "mass":   ak.full_like(data["Muon.PT"][good_mu], 0.10566),
        "charge": data["Muon.Charge"][good_mu],
    }, with_name="Momentum4D")

    all_lep = ak.concatenate([el_vec, mu_vec], axis=1)

    n_lep_pos = ak.sum(all_lep.charge > 0, axis=1)
    n_lep_neg = ak.sum(all_lep.charge < 0, axis=1)

    # Critério Same-Sign (SS): 2 positivos e 0 negativos OU 0 positivos e 2 negativos
    mask_SS  = ((n_lep_pos == 2) & (n_lep_neg == 0)) | \
               ((n_lep_pos == 0) & (n_lep_neg == 2))
    
    mask_III = mask_II & mask_SS
    
    # CORREÇÃO: Faltava calcular o N_III e eff_III
    N_III = ak.sum(mask_III)
    eff_III = float(N_III) / N_total
    print(f"\n  [Corte III] Exatamente 2 léptons SS (pT>10, |eta|<2.5)")
    print(f"              Eventos: {N_III}  |  Eficiência: {eff_III:.4f}")

    # =========================================================
    # CORTE (IV): Massa invariante dilépton M_ll > 100 GeV
    # =========================================================
    # Selecionamos os 2 primeiros léptons de cada evento
    lep_padded = ak.pad_none(all_lep, 2, clip=True, axis=1)
    lep0 = lep_padded[:, 0]
    lep1 = lep_padded[:, 1]

    # MELHORIA: Como usamos o 'vector', a soma lep0 + lep1 resulta em um novo quadrivetor.
    # O método .mass calcula automaticamente a massa invariante, lidando de forma nativa com None.
    mll_vector = (lep0 + lep1).mass
    mll_vector = ak.fill_none(mll_vector, 0.0) # Proteção caso algum evento falhe

    mask_IV = mask_III & (mll_vector > 100.0)
    
    # CORREÇÃO: Faltava calcular o N_IV e eff_IV
    N_IV = ak.sum(mask_IV)
    eff_IV = float(N_IV) / N_total
    print(f"\n  [Corte IV] M_ll > 100 GeV")
    print(f"             Eventos: {N_IV}  |  Eficiência: {eff_IV:.4f}")

    # =========================================================
    # Tabela de eficiências no terminal
    # =========================================================
    print("\n" + "=" * 55)
    print(f"{'Corte':<10} {'Eventos':>10} {'Eficiência':>12}")
    print("-" * 55)
    print(f"{'Total':<10} {N_total:>10}  {'1.000':>10}")
    print(f"{'(I)':<10} {int(N_I):>10}  {eff_I:>10.3f}")
    print(f"{'(II)':<10} {int(N_II):>10}  {eff_II:>10.3f}")
    print(f"{'(III)':<10} {int(N_III):>10}  {eff_III:>10.3f}")
    print(f"{'(IV)':<10} {int(N_IV):>10}  {eff_IV:>10.3f}")
    print("=" * 55)

    return {
        "N_total": N_total,
        "cut_I":   {"N": int(N_I),   "eff": eff_I},
        "cut_II":  {"N": int(N_II),  "eff": eff_II},
        "cut_III": {"N": int(N_III), "eff": eff_III},
        "cut_IV":  {"N": int(N_IV),  "eff": eff_IV},
    }

if __name__ == "__main__":
    input_file = "/home/ellen/Downloads/MG5_aMC_v3_7_1/bin/bttbar/Events/run_01/tag_1_delphes_events.root"
    tree_name  = "Delphes"
    
    # CORREÇÃO: Chamando apenas uma vez e salvando o retorno diretamente
    results = analyze_uproot(input_file, tree_name)

    # Criação do arquivo de log texto com o nome corrigido no print
    output_filename = "cutflow_bttbar.txt"
    with open(output_filename, "w") as f:
        f.write("=======================================================\n")
        f.write("           CUT FLOW ANALYSIS — ttbar Background\n")
        f.write("=======================================================\n\n")

        f.write(f"{'Corte':<10} {'Eventos':>10} {'Eficiência':>12}\n")
        f.write("-------------------------------------------------------\n")
        f.write(f"{'Total':<10} {results['N_total']:>10} {'1.000':>12}\n")
        f.write(f"{'(I)':<10} {results['cut_I']['N']:>10} {results['cut_I']['eff']:>12.4f}\n")
        f.write(f"{'(II)':<10} {results['cut_II']['N']:>10} {results['cut_II']['eff']:>12.4f}\n")
        f.write(f"{'(III)':<10} {results['cut_III']['N']:>10} {results['cut_III']['eff']:>12.4f}\n")
        f.write(f"{'(IV)':<10} {results['cut_IV']['N']:>10} {results['cut_IV']['eff']:>12.4f}\n")
        f.write("=======================================================\n")

    print(f"\nTabela salva em: {output_filename}")