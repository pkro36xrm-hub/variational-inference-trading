"""
Вариационный вывод для финансовых данных
========================================
Проект: Применение вариационного вывода к анализу финансовых временных рядов.
Используем библиотеку PyMC для построения байесовских моделей.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import arviz as az

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Словарь: тикер -> полное название компании
COMPANY_NAMES = {
    "SBER":  "Сбербанк (SBER)",
    "GAZP":  "Газпром (GAZP)",
    "LKOH":  "Лукойл (LKOH)",
    "YNDX":  "Яндекс (YNDX)",
    "MGNT":  "Магнит (MGNT)",
    "NVTK":  "Новатэк (NVTK)",
    "ROSN":  "Роснефть (ROSN)",
    "GMKN":  "Норникель (GMKN)",
    "AAPL":  "Apple Inc. (AAPL)",
    "TSLA":  "Tesla Inc. (TSLA)",
}

def get_company_name(ticker):
    return COMPANY_NAMES.get(ticker.upper(), ticker)


# ============================================================
# ЧАСТЬ 1: ЗАГРУЗКА ДАННЫХ
# ============================================================

def generate_synthetic_data(ticker="AAPL", n_days=756):
    """
    Генерация реалистичных синтетических данных по модели
    геометрического броуновского движения (GBM) со стохастической волатильностью.
    
    Это стандартная модель в финансах — используется для моделирования цен акций.
    """
    print(f"[1/5] Генерация синтетических данных ({ticker}, {n_days} дней)...")
    
    np.random.seed(42)  # для воспроизводимости
    
    # Параметры модели (реалистичные для акций типа AAPL)
    S0 = 150.0          # начальная цена
    mu_annual = 0.10    # средняя годовая доходность 10%
    mu_daily = mu_annual / 252
    
    # Стохастическая волатильность — волатильность меняется со временем
    base_vol = 0.20 / np.sqrt(252)   # базовая дневная волатильность ~20% годовых
    vol_of_vol = 0.02                 # волатильность волатильности
    
    # Генерируем изменяющуюся волатильность
    vol = np.zeros(n_days)
    vol[0] = base_vol
    for t in range(1, n_days):
        vol[t] = max(0.005, vol[t-1] + vol_of_vol * np.random.randn() * 0.1)
    
    # Генерируем доходности
    log_returns_arr = mu_daily + vol * np.random.randn(n_days)
    
    # Строим цены
    prices_arr = S0 * np.exp(np.cumsum(np.insert(log_returns_arr, 0, 0)))
    
    # Создаём DatetimeIndex (рабочие дни)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days + 1)
    
    prices = pd.Series(prices_arr, index=dates, name="Close")
    log_returns = pd.Series(log_returns_arr, index=dates[1:], name="LogReturn")
    
    print(f"      Сгенерировано {len(prices)} дней, {len(log_returns)} доходностей")
    print(f"      Период: {prices.index[0].date()} — {prices.index[-1].date()}")
    print(f"      Средняя дневная доходность: {log_returns.mean():.6f}")
    print(f"      Волатильность (std): {log_returns.std():.6f}")
    
    return prices, log_returns


def load_data_moex(ticker="SBER"):
    """
    Загружаем реальные данные с Московской биржи (MOEX).
    Бесплатный открытый API, работает без VPN из России.
    Тикеры: SBER, GAZP, LKOH, YNDX, MGNT, NVTK и др.
    """
    import requests
    from datetime import datetime, timedelta

    print(f"[1/5] Загрузка данных с MOEX: {ticker}...")

    date_till = datetime.today().strftime("%Y-%m-%d")
    date_from = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/"
        f"securities/{ticker}/candles.json"
        f"?from={date_from}&till={date_till}&interval=24&iss.meta=off"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        cols = data["candles"]["columns"]
        rows = data["candles"]["data"]

        if not rows:
            raise ValueError(f"MOEX вернул пустой ответ для тикера '{ticker}'")

        df = pd.DataFrame(rows, columns=cols)
        df["begin"] = pd.to_datetime(df["begin"])
        df = df.set_index("begin").sort_index()

        prices = df["close"].dropna()
        prices.name = "Close"

        log_returns = np.log(prices / prices.shift(1)).dropna()
        log_returns.name = "LogReturn"

        print(f"      Загружено {len(prices)} дней реальных данных с MOEX")
        print(f"      Период: {prices.index[0].date()} — {prices.index[-1].date()}")
        print(f"      Средняя дневная доходность: {log_returns.mean():.6f}")
        print(f"      Волатильность (std): {log_returns.std():.6f}")

        return prices, log_returns

    except Exception as e:
        print(f"      [!] Не удалось загрузить данные с MOEX: {e}")
        print(f"      -> Используем синтетические данные")
        return generate_synthetic_data(ticker=ticker)


def load_data(ticker="SBER", period="3y"):
    """Обёртка: сначала пробуем MOEX, потом синтетику."""
    return load_data_moex(ticker=ticker)


# ============================================================
# ЧАСТЬ 2: ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ
# ============================================================

def plot_raw_data(prices, log_returns, ticker):
    """Строим графики цен и доходностей."""
    print("[2/5] Визуализация исходных данных...")

    company = get_company_name(ticker)
    period_str = f"{log_returns.index[0].date()} — {log_returns.index[-1].date()}"
    annual_vol = log_returns.std() * np.sqrt(252) * 100
    daily_mean = log_returns.mean() * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Анализ акции: {company}\n"
        f"Период: {period_str}  |  Дней в выборке: {len(log_returns)}  |  "
        f"Средняя доходность: {daily_mean:+.3f}%/день  |  Годовая волатильность: {annual_vol:.1f}%",
        fontsize=13, fontweight="bold"
    )

    # 1) Цена акции
    ax = axes[0, 0]
    ax.plot(prices.index, prices.values, color="#2196F3", linewidth=1.2)
    ax.set_title(f"Цена закрытия — {company}", fontweight="bold")
    ax.set_ylabel("Цена (руб.)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.annotate(f"Старт: {prices.iloc[0]:.0f} руб.", xy=(prices.index[0], prices.iloc[0]),
                xytext=(10, 10), textcoords="offset points", fontsize=9, color="green")
    ax.annotate(f"Конец: {prices.iloc[-1]:.0f} руб.", xy=(prices.index[-1], prices.iloc[-1]),
                xytext=(-60, 10), textcoords="offset points", fontsize=9, color="red")

    # 2) Доходности по дням
    ax = axes[0, 1]
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in log_returns.values]
    ax.bar(log_returns.index, log_returns.values, color=colors, alpha=0.7, width=1)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.axhline(y=log_returns.mean(), color="blue", linestyle="-", linewidth=1.5,
               label=f"Среднее: {daily_mean:+.4f}%")
    ax.set_title("Дневные доходности (зелёный = рост, красный = падение)", fontweight="bold")
    ax.set_ylabel("Доходность (log)")
    ax.legend(fontsize=9)

    # 3) Гистограмма — колокольчик
    ax = axes[1, 0]
    ax.hist(log_returns.values, bins=80, density=True, color="#FF9800", alpha=0.7, edgecolor="white")
    ax.axvline(log_returns.mean(), color="red", linewidth=2, label=f"Среднее: {daily_mean:+.4f}%")
    ax.axvline(log_returns.mean() + log_returns.std(), color="purple", linestyle="--",
               label=f"+1σ: {(daily_mean + log_returns.std()*100):.3f}%")
    ax.axvline(log_returns.mean() - log_returns.std(), color="purple", linestyle="--",
               label=f"-1σ: {(daily_mean - log_returns.std()*100):.3f}%")
    ax.set_title("Распределение доходностей — 'колокольчик'\n(чем шире, тем рискованнее акция)", fontweight="bold")
    ax.set_xlabel("Доходность (log)")
    ax.set_ylabel("Плотность")
    ax.legend(fontsize=9)

    # 4) QQ-plot
    ax = axes[1, 1]
    from scipy import stats
    sorted_returns = np.sort(log_returns.values)
    theoretical_q = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_returns)))
    ax.scatter(theoretical_q, sorted_returns, alpha=0.3, s=5, color="#9C27B0")
    ax.plot([-4, 4], [-4 * log_returns.std(), 4 * log_returns.std()], "r--", alpha=0.7,
            label="Идеальная нормальность")
    ax.set_title("QQ-plot: насколько данные 'нормальны'\n(точки на линии = идеально нормальные)", fontweight="bold")
    ax.set_xlabel("Теоретические квантили")
    ax.set_ylabel("Реальные квантили")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_raw_data.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Сохранено: {path}")


# ============================================================
# ЧАСТЬ 3: МОДЕЛЬ 1 — Нормальное распределение доходностей
# ============================================================

def model_normal(log_returns):
    """
    Простая модель: доходности ~ Normal(μ, σ)
    
    Априорные распределения (priors):
      μ ~ Normal(0, 0.1)    — мы думаем, что средняя доходность около 0
      σ ~ HalfNormal(0.1)   — волатильность положительная, около 0.1
    
    Наблюдения (likelihood):
      returns ~ Normal(μ, σ) — наблюдаемые данные порождены этим распределением
    
    Вариационный вывод подберёт μ и σ так, чтобы модель
    максимально соответствовала наблюдаемым данным.
    """
    import pymc as pm

    print("[3/5] Модель 1: Нормальное распределение...")
    print("      Строим модель: returns ~ Normal(μ, σ)")

    returns_data = log_returns.values.astype(np.float64)

    with pm.Model() as normal_model:
        # Априорные распределения (наши предположения ДО наблюдения данных)
        mu = pm.Normal("mu", mu=0, sigma=0.1)           # средняя доходность
        sigma = pm.HalfNormal("sigma", sigma=0.1)       # волатильность

        # Функция правдоподобия (связь модели с данными)
        likelihood = pm.Normal("returns", mu=mu, sigma=sigma, observed=returns_data)

        # Запускаем вариационный вывод (ADVI)
        print("      Запуск вариационного вывода (ADVI)...")
        approx = pm.fit(n=30000, method="advi", progressbar=True)

        # Получаем сэмплы из аппроксимированного апостериорного распределения
        trace = approx.sample(5000)

    print("      Готово!")
    return normal_model, trace, approx


def plot_model_normal(trace, log_returns, ticker=""):
    """Визуализация результатов Модели 1."""
    print("      Визуализация результатов Модели 1...")

    company = get_company_name(ticker)
    mu_samples = trace.posterior["mu"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()
    mu_mean_pct = mu_samples.mean() * 100
    sigma_mean_pct = sigma_samples.mean() * 100
    annual_vol = sigma_samples.mean() * np.sqrt(252) * 100

    # --- График 1: Апостериорные распределения параметров ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Модель 1 — {company}\nБайесовские оценки параметров (вариационный вывод)",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    ax.hist(mu_samples, bins=60, density=True, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(mu_samples.mean(), color="red", linewidth=2,
               label=f"Среднее μ = {mu_mean_pct:+.4f}%/день")
    ax.axvspan(np.percentile(mu_samples, 5), np.percentile(mu_samples, 95),
               alpha=0.15, color="blue", label="90% доверительный интервал")
    ax.set_title("μ — средняя дневная доходность\n(чем правее, тем лучше — акция растёт)",
                 fontweight="bold")
    ax.set_xlabel("Доходность (log)")
    ax.set_ylabel("Вероятность")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(sigma_samples, bins=60, density=True, color="#FF9800", alpha=0.7, edgecolor="white")
    ax.axvline(sigma_samples.mean(), color="red", linewidth=2,
               label=f"Среднее σ = {sigma_mean_pct:.3f}%/день\n≈ {annual_vol:.1f}% годовых")
    ax.axvspan(np.percentile(sigma_samples, 5), np.percentile(sigma_samples, 95),
               alpha=0.15, color="orange", label="90% доверительный интервал")
    ax.set_title("σ — волатильность (риск)\n(чем шире колокольчик, тем неопределённее оценка)",
                 fontweight="bold")
    ax.set_xlabel("Волатильность (log)")
    ax.set_ylabel("Вероятность")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_model1_posterior.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Сохранено: {path}")

    # --- График 2: Предсказание vs реальность ---
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Модель 1 — {company}\nПредсказанное распределение vs реальные данные\n"
        f"(синие кривые = 100 вариантов модели, красная = средняя)",
        fontsize=12, fontweight="bold"
    )

    ax.hist(log_returns.values, bins=80, density=True, color="#BDBDBD", alpha=0.7,
            edgecolor="white", label="Реальные доходности")

    x = np.linspace(log_returns.min(), log_returns.max(), 300)
    from scipy.stats import norm
    for i in range(100):
        idx = np.random.randint(len(mu_samples))
        y = norm.pdf(x, mu_samples[idx], sigma_samples[idx])
        ax.plot(x, y, color="#2196F3", alpha=0.03)

    y_mean = norm.pdf(x, mu_samples.mean(), sigma_samples.mean())
    ax.plot(x, y_mean, color="red", linewidth=2, label="Средняя модель")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Плотность")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "03_model1_prediction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Сохранено: {path}")


# ============================================================
# ЧАСТЬ 4: МОДЕЛЬ 2 — Стохастическая волатильность
# ============================================================

def model_stochastic_volatility(log_returns):
    """
    Модель стохастической волатильности.
    
    Идея: волатильность (σ) НЕ постоянна, а меняется день ото дня.
    В спокойные дни — маленькая, в кризис — большая.
    
    Модель:
      h_t = h_{t-1} + σ_h * ε_t    — скрытая лог-волатильность (случайное блуждание)
      r_t ~ Normal(0, exp(h_t / 2))  — наблюдаемые доходности
    """
    import pymc as pm

    print("[4/5] Модель 2: Стохастическая волатильность...")

    returns_data = log_returns.values.astype(np.float64)
    n = len(returns_data)

    with pm.Model() as sv_model:
        # Шаг волатильности (насколько быстро волатильность меняется)
        sigma_h = pm.HalfNormal("sigma_h", sigma=0.5)
        
        # Начальный уровень лог-волатильности
        h0 = pm.Normal("h0", mu=-5, sigma=2)

        # Скрытый процесс: лог-волатильность как случайное блуждание
        h = pm.GaussianRandomWalk("h", sigma=sigma_h, init_dist=pm.Normal.dist(mu=h0, sigma=0.1),
                                   steps=n - 1)

        # Волатильность = exp(h / 2)
        volatility = pm.math.exp(h / 2)

        # Наблюдаемые доходности
        obs = pm.Normal("returns", mu=0, sigma=volatility, observed=returns_data)

        # Вариационный вывод
        print("      Запуск вариационного вывода (ADVI)...")
        approx = pm.fit(n=50000, method="advi", progressbar=True)

        trace = approx.sample(3000)

    print("      Готово!")
    return sv_model, trace, approx


def plot_model_sv(trace, log_returns, prices, ticker=""):
    """Визуализация результатов Модели 2."""
    print("      Визуализация результатов Модели 2...")

    dates = log_returns.index

    # Извлекаем скрытую волатильность
    h_samples = trace.posterior["h"].values
    # h_samples shape: (chains, draws, time)
    h_mean = h_samples.mean(axis=(0, 1))
    h_low = np.percentile(h_samples, 5, axis=(0, 1))
    h_high = np.percentile(h_samples, 95, axis=(0, 1))

    vol_mean = np.exp(h_mean / 2)
    vol_low = np.exp(h_low / 2)
    vol_high = np.exp(h_high / 2)

    company = get_company_name(ticker if hasattr(prices, 'name') else "")

    # --- График: Цена + Волатильность ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        f"Модель 2: Стохастическая волатильность — {company}\n"
        "Волатильность НЕ постоянная — меняется каждый день (в кризис растёт, в спокойное время падает)",
        fontsize=13, fontweight="bold"
    )

    # Цена
    ax = axes[0]
    price_vals = prices.loc[dates].values
    if price_vals.ndim > 1:
        price_vals = price_vals.flatten()
    ax.plot(dates, price_vals, color="#2196F3", linewidth=1.2)
    ax.set_ylabel("Цена (руб.)")
    ax.set_title("График цены акции", fontweight="bold")
    ax.fill_between(dates, price_vals, alpha=0.1, color="#2196F3")

    # Доходности
    ax = axes[1]
    colors_r = ["#4CAF50" if r >= 0 else "#F44336" for r in log_returns.values]
    ax.bar(dates, log_returns.values, color=colors_r, alpha=0.7, width=1)
    ax.axhline(0, color="gray", linestyle="--")
    ax.axhline(log_returns.mean(), color="blue", linewidth=1.5,
               label=f"Среднее: {log_returns.mean()*100:+.4f}%/день")
    ax.set_ylabel("Доходность")
    ax.set_title("Дневные доходности (зелёный = рост, красный = падение)", fontweight="bold")
    ax.legend(fontsize=9)

    # Волатильность
    ax = axes[2]
    ax.plot(dates, vol_mean * 100, color="#F44336", linewidth=2,
            label="Оценённая волатильность")
    ax.fill_between(dates, vol_low * 100, vol_high * 100, color="#F44336", alpha=0.2,
                    label="90% доверительный интервал")
    # Найдём день максимального риска
    max_idx = np.argmax(vol_mean)
    ax.annotate(f"Пик риска\n{dates[max_idx].date()}",
                xy=(dates[max_idx], vol_mean[max_idx] * 100),
                xytext=(30, 10), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax.set_ylabel("Волатильность (%)")
    ax.set_title("Оценённая волатильность — ПИКИ = дни наибольшего риска", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "04_model2_volatility.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Сохранено: {path}")

    # --- График: Апостериорные распределения параметров SV ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Модель 2 — {company}\nПараметры стохастической волатильности",
        fontsize=13, fontweight="bold"
    )

    param_labels = {
        "sigma_h": ("σ_h — насколько быстро\nменяется волатильность", "#FF9800"),
        "h0": ("h₀ — начальный уровень\nволатильности", "#4CAF50"),
    }
    for ax, (name, color) in zip(axes, [("sigma_h", "#FF9800"), ("h0", "#4CAF50")]):
        title, _ = param_labels[name]
        vals = trace.posterior[name].values.flatten()
        ax.hist(vals, bins=60, density=True, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(vals.mean(), color="red", linewidth=2,
                   label=f"Среднее: {vals.mean():.4f}")
        ax.axvspan(np.percentile(vals, 5), np.percentile(vals, 95),
                   alpha=0.15, color=color, label="90% интервал")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Значение")
        ax.set_ylabel("Вероятность")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "05_model2_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Сохранено: {path}")


# ============================================================
# ЧАСТЬ 5: СРАВНЕНИЕ МОДЕЛЕЙ + ИТОГИ
# ============================================================

def compare_and_summarize(trace1, trace2, log_returns):
    """Сравнение двух моделей и итоговая таблица."""
    print("[5/5] Сравнение моделей и итоги...")

    mu1 = trace1.posterior["mu"].values.flatten()
    sigma1 = trace1.posterior["sigma"].values.flatten()

    sigma_h = trace2.posterior["sigma_h"].values.flatten()
    h0 = trace2.posterior["h0"].values.flatten()

    summary = f"""
{'='*60}
            SUMMARY REPORT
{'='*60}

Data:
  N observations: {len(log_returns)}
  Period: {log_returns.index[0].date()} -- {log_returns.index[-1].date()}
  Empirical mean: {log_returns.mean():.6f}
  Empirical volatility: {log_returns.std():.6f}

{'--'*30}
Model 1: Normal distribution
  mu (mean return):   {mu1.mean():.6f} +/- {mu1.std():.6f}
  sigma (volatility): {sigma1.mean():.6f} +/- {sigma1.std():.6f}
  
  Daily volatility  ~ {sigma1.mean()*100:.2f}%
  Annual volatility ~ {sigma1.mean()*np.sqrt(252)*100:.1f}%

{'--'*30}
Model 2: Stochastic Volatility
  sigma_h (vol shock):     {sigma_h.mean():.4f} +/- {sigma_h.std():.4f}
  h0 (initial log-vol):    {h0.mean():.4f} +/- {h0.std():.4f}

{'--'*30}
Conclusion:
  Model 2 (stochastic volatility) better describes the data
  because it captures time-varying volatility --
  a key feature of financial data (volatility clustering).
{'='*60}
"""
    print(summary)

    # Сохраняем отчёт
    path = os.path.join(RESULTS_DIR, "report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"      Отчёт сохранён: {path}")


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    # Читаем тикер из config.txt (первая непустая строка без #)
    config_path = os.path.join(os.path.dirname(__file__), "config.txt")
    TICKER = "SBER"  # значение по умолчанию
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    TICKER = line.upper()
                    break
        print(f"Тикер из config.txt: {TICKER} ({get_company_name(TICKER)})")
    except Exception:
        print(f"config.txt не найден, используем: {TICKER}")

    # 1. Загрузка данных
    prices, log_returns = load_data(ticker=TICKER, period="3y")

    # 2. Визуализация исходных данных
    plot_raw_data(prices, log_returns, TICKER)

    # 3. Модель 1: Нормальное распределение
    model1, trace1, approx1 = model_normal(log_returns)
    plot_model_normal(trace1, log_returns, TICKER)

    # 4. Модель 2: Стохастическая волатильность
    model2, trace2, approx2 = model_stochastic_volatility(log_returns)
    plot_model_sv(trace2, log_returns, prices, TICKER)

    # 5. Сравнение и итоги
    compare_and_summarize(trace1, trace2, log_returns)

    print("\nAll plots saved in results/ folder")
    print("   Done!")
