import re
import time
import random
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright
from scipy.stats import poisson, nbinom
from tabulate import tabulate

# --- CONFIGURAÇÕES ---
COMPETICAO_KEYWORD = "brasileir"
URL_JOGO = (
    "https://www.sofascore.com/football/match/sao-paulo-fluminense/lOsGO#id:13472605"
)
NUM_JOGOS_ANALISE = 10


# --- CORES PARA O TERMINAL ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"  # Para Over seguro
    RED = "\033[91m"  # Para Alerta
    CYAN = "\033[96m"  # Para Under
    YELLOW = "\033[93m"  # Para Destaque


def extrair_valor_universal(groups, keywords, is_home):
    if not groups:
        return 0
    keywords = [k.lower() for k in keywords]
    for g in groups:
        for item in g["statisticsItems"]:
            item_name = item["name"].lower()
            if any(k in item_name for k in keywords):
                try:
                    val = item["home"] if is_home else item["away"]
                    return float(val)
                except:
                    continue
    return 0


def get_stats_avancadas(page, event_id, is_home):
    script_stats = f"""
        async () => {{
            try {{
                const r = await fetch('https://www.sofascore.com/api/v1/event/{event_id}/statistics');
                if (r.status !== 200) return null;
                return await r.json();
            }} catch {{ return null; }}
        }}
    """
    time.sleep(random.uniform(0.1, 0.3))
    raw = page.evaluate(script_stats)

    stats = {
        "corners": 0,
        "corners_ht": 0,
        "corners_2t": 0,
        "shots_ot_ht": 0,
        "shots_ot_2t": 0,
        "shots_tot_ht": 0,
        "shots_tot_2t": 0,
    }

    if raw and "statistics" in raw:
        stats_all = next(
            (p["groups"] for p in raw["statistics"] if p["period"] == "ALL"), []
        )
        stats_1st = next(
            (p["groups"] for p in raw["statistics"] if p["period"] == "1ST"), []
        )
        stats_2nd = next(
            (p["groups"] for p in raw["statistics"] if p["period"] == "2ND"), []
        )

        stats["corners"] = int(
            extrair_valor_universal(stats_all, ["corner", "escanteio"], is_home)
        )
        stats["corners_ht"] = int(
            extrair_valor_universal(stats_1st, ["corner", "escanteio"], is_home)
        )
        stats["corners_2t"] = int(
            extrair_valor_universal(stats_2nd, ["corner", "escanteio"], is_home)
        )

        if stats["corners_2t"] == 0 and stats["corners"] > 0:
            stats["corners_2t"] = stats["corners"] - stats["corners_ht"]

        stats["shots_ot_ht"] = int(
            extrair_valor_universal(
                stats_1st, ["shots on target", "chutes no gol"], is_home
            )
        )
        stats["shots_ot_2t"] = int(
            extrair_valor_universal(
                stats_2nd, ["shots on target", "chutes no gol"], is_home
            )
        )

    return stats


def processar_time(page, team_id, team_name, is_home_mode):
    print(
        f"\n🔰 COLETANDO DADOS: {Colors.BOLD}{team_name.upper()}{Colors.RESET} ({'CASA' if is_home_mode else 'FORA'})"
    )
    # Cabeçalho ajustado com PLACAR
    print(
        f"{'DATA':<6} | {'PLACAR':<6} | {'OPONENTE':<12} | {'HT (C/S)':<8} | {'2T (C/S)':<8} | {'FT (C/S)':<8}"
    )
    print("-" * 75)

    jogos = []

    for pag in range(4):
        try:
            script = f"async () => (await fetch('https://www.sofascore.com/api/v1/team/{team_id}/events/last/{pag}')).json()"
            resp = page.evaluate(script)
        except:
            continue

        if not resp or "events" not in resp:
            break

        for e in resp["events"]:
            if len(jogos) >= NUM_JOGOS_ANALISE:
                break
            if e["status"]["type"] != "finished":
                continue

            tourn_name = e["tournament"]["uniqueTournament"]["name"].lower()
            if COMPETICAO_KEYWORD not in tourn_name:
                continue

            sou_mandante = e["homeTeam"]["id"] == team_id
            if is_home_mode != sou_mandante:
                continue

            # Extração do Placar
            try:
                placar_h = e["homeScore"]["display"]
                placar_a = e["awayScore"]["display"]
                placar_final = f"{placar_h}-{placar_a}"
            except:
                placar_final = "?-?"

            stats = get_stats_avancadas(page, e["id"], sou_mandante)

            row = {
                "data": datetime.fromtimestamp(int(e["startTimestamp"])).strftime(
                    "%d/%m"
                ),
                "placar": placar_final,
                "oponente": (
                    e["awayTeam"]["name"] if is_home_mode else e["homeTeam"]["name"]
                ),
                "cantos_ht": stats["corners_ht"],
                "cantos_2t": stats["corners_2t"],
                "cantos_ft": stats["corners"],
                "chutes_ht": stats["shots_ot_ht"],
                "chutes_2t": stats["shots_ot_2t"],
                "chutes_ft": stats["shots_ot_ht"] + stats["shots_ot_2t"],
            }
            jogos.append(row)

            # Print formatado
            op_curto = (
                (row["oponente"][:10] + "..")
                if len(row["oponente"]) > 10
                else row["oponente"]
            )
            ht_view = f"{row['cantos_ht']}/{row['chutes_ht']}"
            st_view = f"{row['cantos_2t']}/{row['chutes_2t']}"
            ft_view = f"{row['cantos_ft']}/{row['chutes_ft']}"

            # Destaque de cor se tiver muitos escanteios
            if row["cantos_ft"] >= 10:
                ft_view = f"{Colors.YELLOW}{ft_view}{Colors.RESET}"

            print(
                f"{row['data']:<6} | {row['placar']:<6} | {op_curto:<12} | {ht_view:<8} | {st_view:<8} | {ft_view:<8}"
            )

    return pd.DataFrame(jogos)


def analise_quantitativa(df_home, df_away):
    if df_home.empty or df_away.empty:
        print(f"\n{Colors.RED}❌ ERRO: Dados insuficientes.{Colors.RESET}")
        return

    # --- ENGENHARIA ESTATÍSTICA ---
    mercados = [
        {
            "nome": "JOGO COMPLETO",
            "df_h": df_home["cantos_ft"],
            "df_a": df_away["cantos_ft"],
            "linhas": [8.5, 9.5, 10.5, 11.5, 12.5],
        },
        {
            "nome": "MANDANTE",
            "df_h": df_home["cantos_ft"],
            "df_a": None,
            "linhas": [4.5, 5.5, 6.5],
        },
        {
            "nome": "VISITANTE",
            "df_h": None,
            "df_a": df_away["cantos_ft"],
            "linhas": [3.5, 4.5, 5.5],
        },
        {
            "nome": "1º TEMPO (HT)",
            "df_h": df_home["cantos_ht"],
            "df_a": df_away["cantos_ht"],
            "linhas": [3.5, 4.5, 5.5],
        },
        {
            "nome": "2º TEMPO (FT)",
            "df_h": df_home["cantos_2t"],
            "df_a": df_away["cantos_2t"],
            "linhas": [3.5, 4.5, 5.5],
        },
    ]

    oportunidades = []

    print("\n" + "▓" * 80)
    print(f" 🧠 CÉREBRO ESTATÍSTICO (Com Monte Carlo)")
    print("▓" * 80)

    for m in mercados:
        lambdas = []
        vars_val = []

        for df in [m["df_h"], m["df_a"]]:
            if df is not None:
                mean_10 = df.mean()
                mean_5 = df.head(5).mean()
                l_ajustado = (mean_10 * 0.6) + (mean_5 * 0.4)
                lambdas.append(l_ajustado)
                vars_val.append(df.var())

        lambda_final = sum(lambdas)
        var_final = sum(vars_val)
        if len(lambdas) == 1:
            var_final = vars_val[0]

        # Bonus Pressão
        if "1º TEMPO" in m["nome"]:
            proj_chutes = df_home["chutes_ht"].mean() + df_away["chutes_ht"].mean()
            if proj_chutes > 8:
                lambda_final *= 1.1

        usar_negbin = False
        if var_final > lambda_final:
            usar_negbin = True
            p_nb = lambda_final / var_final
            n_nb = (lambda_final**2) / (var_final - lambda_final)

        # MONTE CARLO (10k)
        if usar_negbin:
            simulacoes = nbinom.rvs(n_nb, p_nb, size=10000)
        else:
            simulacoes = poisson.rvs(lambda_final, size=10000)

        for linha in m["linhas"]:
            # --- OVER ---
            prob_mc_over = (simulacoes > linha).mean()
            prob_final_over = prob_mc_over  # Focando no Monte Carlo direto
            odd_justa_over = 1 / prob_final_over if prob_final_over > 0 else 99

            if 1.45 <= odd_justa_over <= 2.60:
                cv = (var_final**0.5) / lambda_final if lambda_final > 0 else 1
                score = prob_final_over * (1 - (cv * 0.3))
                oportunidades.append(
                    {
                        "Mercado": m["nome"],
                        "Seleção": f"Over {linha}",
                        "Prob": prob_final_over,
                        "Odd": odd_justa_over,
                        "Score": score,
                        "Tipo": "OVER",
                    }
                )

            # --- UNDER ---
            prob_mc_under = (simulacoes <= linha).mean()
            prob_final_under = prob_mc_under
            odd_justa_under = 1 / prob_final_under if prob_final_under > 0 else 99

            if 1.45 <= odd_justa_under <= 2.10:
                cv = (var_final**0.5) / lambda_final if lambda_final > 0 else 1
                score = prob_final_under * (1 - (cv * 0.5))
                oportunidades.append(
                    {
                        "Mercado": m["nome"],
                        "Seleção": f"Under {linha}",
                        "Prob": prob_final_under,
                        "Odd": odd_justa_under,
                        "Score": score,
                        "Tipo": "UNDER",
                    }
                )

    oportunidades.sort(key=lambda x: x["Score"], reverse=True)
    top_picks = oportunidades[:7]

    # --- TABELA FORMATADA (TOP 7) ---
    print(f"\n🏆 {Colors.BOLD}TOP 7 OPORTUNIDADES (DATA DRIVEN){Colors.RESET}")

    tabela_display = []
    for pick in top_picks:
        # Formatação Condicional de Cor
        if pick["Tipo"] == "OVER":
            cor = Colors.GREEN
            seta = "▲"
        else:
            cor = Colors.CYAN
            seta = "▼"

        # Monta a linha colorida para o Tabulate
        mercado_fmt = f"{pick['Mercado']}"
        linha_fmt = f"{cor}{pick['Seleção']}{Colors.RESET}"
        prob_fmt = f"{pick['Prob'] * 100:.1f}%"
        odd_fmt = f"{Colors.BOLD}@{pick['Odd']:.2f}{Colors.RESET}"
        direcao_fmt = f"{cor}{seta} {pick['Tipo']}{Colors.RESET}"

        tabela_display.append([mercado_fmt, linha_fmt, prob_fmt, odd_fmt, direcao_fmt])

    # Headers limpos
    headers = ["MERCADO", "LINHA", "PROB.", "ODD JUSTA", "TIPO"]

    # Renderiza tabela estilo 'fancy_grid'
    print(
        tabulate(
            tabela_display, headers=headers, tablefmt="fancy_grid", stralign="center"
        )
    )


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            }
        )

        print(f"{Colors.YELLOW}🌐 Conectando ao Sofascore...{Colors.RESET}")
        try:
            match_id = re.search(r"id:(\d+)", URL_JOGO).group(1)
        except AttributeError:
            return
        page.goto(URL_JOGO)
        time.sleep(3)
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev = page.evaluate(f"async () => (await fetch('{api_url}')).json()")
        if not ev or "event" not in ev:
            return
        ev = ev["event"]

        df_h = processar_time(page, ev["homeTeam"]["id"], ev["homeTeam"]["name"], True)
        df_a = processar_time(page, ev["awayTeam"]["id"], ev["awayTeam"]["name"], False)
        browser.close()
        analise_quantitativa(df_h, df_a)


if __name__ == "__main__":
    main()
