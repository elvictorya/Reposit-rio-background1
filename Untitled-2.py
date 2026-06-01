"""
cut_analysis_WWjj.py
====================
Aplica os cortes cinemáticos sequenciais do artigo (Eq. 5) ao arquivo ROOT
gerado pelo Delphes para o processo pp > jj W+ W+ (W+ > l+ vl).

Cortes aplicados (SS leptons, última coluna da Tabela I):
  (I)   Pelo menos 1 b-jet com pT > 20 GeV e |eta| < 2.5
  (II)  MET > 20 GeV
  (III) Exatamente 2 léptons SS com pT > 10 GeV e |eta| < 2.5
  (IV)  Massa invariante dilépton M_ll > 100 GeV

Uso:
    python cut_analysis_WWjj.py --input /caminho/para/tag_1_delphes_events.root
    python cut_analysis_WWjj.py --input Events/run_01/tag_1_delphes_events.root

Dependências:
    pip install uproot awkward vector numpy
    (ou use ROOT/PyROOT se preferir)
"""

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import uproot
    import awkward as ak
    import vector
    vector.register_awkward()  #vector.register_awkward(): registra os quadrivetores dentro do awkward, permitindo 
    #fazer lep0 + lep1 e .mass diretamente em arrays awkward
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


# =============================================================================
# Análise via uproot + awkward (recomendado)
# =============================================================================

#leitura dos dados 
def analyze_uproot(input_file: str, tree_name: str = "Delphes"):
    print(f"\n[uproot] Lendo: {input_file}  (tree: {tree_name})")
    f = uproot.open(input_file)
    tree = f[tree_name]

    # ---- Leitura dos branches necessários ---- (em vez de ler branch por branch, lemos tudo de uma vez passando uma 
    #lista de branches. O resultado é um único "dicionário de arrays" chamado data, onde acessamos fazendo data["Jet.PT"], data["Electron.Eta"], etc.)
    branches = [
        # Jets
        "Jet.PT", "Jet.Eta", "Jet.BTag",
        # MET
        "MissingET.MET",
        # Elétrons
        "Electron.PT", "Electron.Eta", "Electron.Charge",
        # Múons
        "Muon.PT", "Muon.Eta", "Muon.Charge",
    ]

    data = tree.arrays(branches, library="ak")

    N_total = len(data)
    print(f"  Eventos totais: {N_total}")

    # =========================================================
    # CORTE (I): >= 1 b-jet com pT > 20 GeV e |eta| < 2.5
    #fazemos variável = data[branch], ou seja, data["Jet.PT"] é um array com os valores de pT dos jets para todos os eventos. O mesmo para eta e btag.
    # =========================================================
    jet_pt   = data["Jet.PT"]
    jet_eta  = data["Jet.Eta"]
    jet_btag = data["Jet.BTag"]  # 1 se b-tagged, 0 se não

#filtro:
    good_bjet = (jet_pt > 20.0) & (np.abs(jet_eta) < 2.5) & (jet_btag == 1)
    n_bjets   = ak.sum(good_bjet, axis=1)  #axis=1 significa somar dentro de cada evento
    mask_I    = n_bjets >= 1
#contanto os b-jets que passaram pelo cutI
    N_I = ak.sum(mask_I)
    eff_I = float(N_I) / N_total
    print(f"\n  [Corte I]  >= 1 b-jet (pT>20, |eta|<2.5)")
    print(f"             Eventos: {N_I}  |  Eficiência: {eff_I:.4f}")

    # =========================================================
    # CORTE (II): MET > 20 GeV
    # =========================================================
    met      = data["MissingET.MET"]
    # MissingET é uma leaf com 1 valor por evento (array de tamanho 1)
    met_val  = ak.firsts(met, axis=1)  # pega o primeiro (e único) valor
    met_val  = ak.fill_none(met_val, 0.0)  # preenche com 0.0 se não houver MET (eventos sem MissingET)

    mask_II  = mask_I & (met_val > 20.0)
    N_II     = ak.sum(mask_II)
    eff_II   = float(N_II) / N_total
    print(f"\n  [Corte II] MET > 20 GeV")
    print(f"             Eventos: {N_II}  |  Eficiência: {eff_II:.4f}")

    # =========================================================
    # CORTE (III): Exatamente 2 léptons SS com pT > 10 GeV, |eta| < 2.5 (elétrons e múons)
    # Para W+W+: esperamos 2 léptons com carga +1
    # =========================================================
    el_pt     = data["Electron.PT"]
    el_eta    = data["Electron.Eta"]
    el_charge = data["Electron.Charge"]

    mu_pt     = data["Muon.PT"]
    mu_eta    = data["Muon.Eta"]
    mu_charge = data["Muon.Charge"]

#selecionando os elétrons e o múons que satisfazem as condições do cut3 (pT > 10 GeV e |eta| < 2.5)
    good_el = (el_pt > 10.0) & (np.abs(el_eta) < 2.5)
    good_mu = (mu_pt > 10.0) & (np.abs(mu_eta) < 2.5)

    # Cargas dos léptons selecionados(good_el e good_mu)
    el_sel_charge = el_charge[good_el]
    mu_sel_charge = mu_charge[good_mu]

    # Número de léptons positivos e negativos
    n_el_pos = ak.sum(el_sel_charge > 0, axis=1)
    n_el_neg = ak.sum(el_sel_charge < 0, axis=1)
    n_mu_pos = ak.sum(mu_sel_charge > 0, axis=1)
    n_mu_neg = ak.sum(mu_sel_charge < 0, axis=1)

    n_lep_pos = n_el_pos + n_mu_pos  # total de léptons +
    n_lep_neg = n_el_neg + n_mu_neg  # total de léptons -
    n_lep_tot = n_lep_pos + n_lep_neg

    # W+W+: exatamente 2 léptons, ambos com carga + (SS)
    # W-W-: exatamente 2 léptons, ambos com carga - (SS)
    mask_SS = ((n_lep_pos == 2) & (n_lep_neg == 0)) | \
              ((n_lep_pos == 0) & (n_lep_neg == 2))
    mask_III = mask_II & mask_SS

    N_III    = ak.sum(mask_III)
    eff_III  = float(N_III) / N_total
    print(f"\n  [Corte III] Exatamente 2 léptons SS (pT>10, |eta|<2.5)")
    print(f"              Eventos: {N_III}  |  Eficiência: {eff_III:.4f}")

    # =========================================================
    # CORTE (IV): Massa invariante dilépton M_ll > 100 GeV
    # =========================================================
    # Reconstruímos os quadrivetores dos léptons selecionados
    # Elétrons (massa ~ 0.000511 GeV, usamos 0)
    el_px = el_pt * np.cos(np.arctan2(np.sin(el_eta * 0), 1))  # simplificado
    # Usamos a abordagem com vector para cálculo correto

    # Quadrivetores usando apenas pT, eta, phi=0 (Delphes fornece phi também,
    # mas para M_ll pT e eta são suficientes se phi vier junto)
    # Como não carregamos phi acima, vamos incluí-lo:
    print("\n  [Aviso] Para M_ll preciso carregar Electron.Phi e Muon.Phi...")

    f2 = uproot.open(input_file)
    t2 = f2[tree_name]
    extra = t2.arrays(
        ["Electron.Phi", "Muon.Phi"],
        library="ak"
    )
    el_phi = extra["Electron.Phi"]
    mu_phi = extra["Muon.Phi"]

    # Elétrons bons
    el_vec = ak.zip({
        "pt":  el_pt[good_el],
        "eta": data["Electron.Eta"][good_el],
        "phi": el_phi[good_el],
        "mass": ak.zeros_like(el_pt[good_el]),
    }, with_name="Momentum4D")

    # Múons bons
    mu_vec = ak.zip({
        "pt":  mu_pt[good_mu],
        "eta": data["Muon.Eta"][good_mu],
        "phi": mu_phi[good_mu],
        "mass": ak.full_like(mu_pt[good_mu], 0.10566),  # massa do múon em GeV
    }, with_name="Momentum4D")

    # Combina léptons de todos os sabores
    all_lep = ak.concatenate([el_vec, mu_vec], axis=1)

    # Para eventos que passaram o corte III, pega os 2 léptons e calcula M_ll
    # (já sabemos que são exatamente 2 SS)
    lep0 = ak.firsts(all_lep, axis=1)
    lep1 = ak.pad_none(all_lep, 2, axis=1)[:, 1]

    # Soma quadrivetores
    m_ll = ak.fill_none(
        (lep0 + lep1).mass if (ak.count(ak.is_none(lep0)) == 0) else ak.Array([0.0]),
        0.0
    )

    # Cálculo mais robusto evento a evento:
    def calc_mll(leps):
        """Calcula M_ll para eventos com exatamente 2 léptons."""
        # Retorna ak array com M_ll
        l0 = leps[:, 0:1]  # primeiro lépton
        l1 = leps[:, 1:2]  # segundo lépton
        v0 = ak.firsts(l0)
        v1 = ak.firsts(l1)
        combined = v0 + v1
        return ak.fill_none(combined.mass, 0.0)

    # Aplica apenas nos eventos com exatamente 2 léptons bons
    has_2lep = ak.num(all_lep, axis=1) == 2
    m_ll_vals = ak.where(
        has_2lep,
        calc_mll(ak.pad_none(all_lep, 2, clip=True)),
        ak.zeros_like(met_val)
    )
    # calc_mll com pad_none retorna array com None → precisamos unwrap
    # Abordagem alternativa mais simples:
    # Pegar pT, eta, phi dos 2 primeiros léptons e calcular manualmente

    lep_pt  = ak.pad_none(all_lep.pt,  2, clip=True, axis=1)
    lep_eta = ak.pad_none(all_lep.eta, 2, clip=True, axis=1)
    lep_phi = ak.pad_none(all_lep.phi, 2, clip=True, axis=1)

    pt0  = ak.fill_none(lep_pt[:, 0],  0.0)
    pt1  = ak.fill_none(lep_pt[:, 1],  0.0)
    eta0 = ak.fill_none(lep_eta[:, 0], 0.0)
    eta1 = ak.fill_none(lep_eta[:, 1], 0.0)
    phi0 = ak.fill_none(lep_phi[:, 0], 0.0)
    phi1 = ak.fill_none(lep_phi[:, 1], 0.0)

    # Quadrivetores cartesianos (massa ~ 0 para léptons relativísticos)
    px0 = pt0 * np.cos(phi0);  py0 = pt0 * np.sin(phi0)
    pz0 = pt0 * np.sinh(eta0); E0  = pt0 * np.cosh(eta0)

    px1 = pt1 * np.cos(phi1);  py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1); E1  = pt1 * np.cosh(eta1)

    E_ll   = E0 + E1
    px_ll  = px0 + px1
    py_ll  = py0 + py1
    pz_ll  = pz0 + pz1
    M2_ll  = E_ll**2 - px_ll**2 - py_ll**2 - pz_ll**2
    M_ll   = np.sqrt(np.maximum(M2_ll, 0.0))

    mask_IV  = mask_III & (M_ll > 100.0)
    N_IV     = ak.sum(mask_IV)
    eff_IV   = float(N_IV) / N_total
    print(f"\n  [Corte IV] M_ll > 100 GeV")
    print(f"             Eventos: {N_IV}  |  Eficiência: {eff_IV:.4f}")

    # =========================================================
    # Tabela de eficiências
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
    #print("\nValores esperados (Tabela I, W±W±jj):")
    #print("  (I) = 0.11 | (II) = 0.11 | (III) = 0.03 | (IV) = 0.02")

    return {
        "N_total": N_total,
        "cut_I":   {"N": int(N_I),   "eff": eff_I},
        "cut_II":  {"N": int(N_II),  "eff": eff_II},
        "cut_III": {"N": int(N_III), "eff": eff_III},
        "cut_IV":  {"N": int(N_IV),  "eff": eff_IV},
    }

if __name__ == "__main__":
    input_file = "/home/ellen/Downloads/MG5_aMC_v3_7_1/bin/background1/Events/run_01/tag_1_delphes_events.root"
    tree_name  = "Delphes"
    analyze_uproot(input_file, tree_name)

    # salva os resultados retornados pela função
    results = analyze_uproot(input_file, tree_name)

    # cria arquivo texto
    with open("cutflow(background1).txt", "w") as f:

        f.write("=======================================================\n")
        f.write("           CUT FLOW ANALYSIS — W±W±jj\n")
        f.write("=======================================================\n\n")

        f.write(f"{'Corte':<10} {'Eventos':>10} {'Eficiência':>12}\n")
        f.write("-------------------------------------------------------\n")

        f.write(f"{'Total':<10} {results['N_total']:>10} {'1.000':>12}\n")

        f.write(
            f"{'(I)':<10} "
            f"{results['cut_I']['N']:>10} "
            f"{results['cut_I']['eff']:>12.4f}\n"
        )

        f.write(
            f"{'(II)':<10} "
            f"{results['cut_II']['N']:>10} "
            f"{results['cut_II']['eff']:>12.4f}\n"
        )

        f.write(
            f"{'(III)':<10} "
            f"{results['cut_III']['N']:>10} "
            f"{results['cut_III']['eff']:>12.4f}\n"
        )

        f.write(
            f"{'(IV)':<10} "
            f"{results['cut_IV']['N']:>10} "
            f"{results['cut_IV']['eff']:>12.4f}\n"
        )

        f.write("=======================================================\n")

    print("\nTabela salva em: cutflow(background1).txt")
