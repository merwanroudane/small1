import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Small Sample Effects in Econometrics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom function to create downloadable plots
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    "Introduction & Definitions",
    "Visualization of Sample Size Effects",
    "Hypothesis Testing & Power",
    "Coefficient Estimation & Bias",
    "Time Series Challenges",
    "Model Selection Issues",
    "Regularization Solutions",
    "Bootstrap & Resampling",
    "Bayesian Approaches",
    "Practical Guidelines"
]
selected_page = st.sidebar.radio("Go to", pages)

# Add sidebar information
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This interactive app demonstrates the effects of small sample sizes in econometric analysis, "
    "providing visualizations and explanations of key challenges and solutions."
)
st.sidebar.markdown("---")
st.sidebar.header("Resources")
st.sidebar.markdown(
    """
    - [Wooldridge's Econometric Analysis](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/9781337558860/)
    - [Stock & Watson Introductory Econometrics](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000003500)
    - [Mostly Harmless Econometrics](https://press.princeton.edu/books/paperback/9780691120355/mostly-harmless-econometrics)
    """
)

###########################################
# PAGE: Introduction & Definitions
###########################################
if selected_page == "Introduction & Definitions":
    st.title("Small Sample Effects in Econometrics: Introduction & Definitions")

    st.markdown("""
    ## What are "Small Samples" in Econometrics?

    In econometrics, the definition of a "small sample" isn't universal but generally refers to datasets where asymptotic 
    (large-sample) properties of estimators may not apply reliably. This interactive application explores the challenges
    posed by small samples and provides techniques to address them.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Common Definitions of Small Samples

        - **Absolute size**: Typically n < 30 observations
        - **Relative to parameters**: n/k < 10 (where k is the number of parameters)
        - **Time series context**: Small T < 50 periods
        - **Panel data context**: Small N (cross-sections), small T (time periods), or both
        """)

        st.info("""
        **Why Small Samples Matter**

        Most econometric theory relies on asymptotic properties that only apply as sample sizes approach infinity.
        In practice, economists often work with limited data, especially in:

        - Macroeconomic analysis (limited time periods)
        - Development economics (small-scale interventions)
        - Industrial organization (few firms in a market)
        - Finance (rare events, crisis periods)
        - Specialized labor markets
        """)

    with col2:
        st.markdown("""
        ### Asymptotic vs. Finite Sample Properties

        | Property | Large Sample (Asymptotic) | Small Sample (Finite) |
        |----------|--------------------------|----------------------|
        | **Estimator bias** | Focus on consistency | Exact bias matters |
        | **Distributions** | Normal approximations | Exact distributions |
        | **Inference** | Z-tests | t-tests with df adjustments |
        | **Efficiency** | Asymptotic efficiency | Exact efficiency |
        | **Hypothesis tests** | Nominal size approximately correct | Size distortions common |
        | **Confidence intervals** | Based on normal approximation | Need exact methods |
        """)

        st.warning("""
        **Statistical vs. Practical Significance**

        Small samples make it harder to achieve statistical significance even when effects are economically meaningful.
        Focus on:
        - Effect sizes
        - Confidence intervals
        - Power limitations
        - Economic significance
        """)

    st.markdown("---")

    st.markdown("""
    ## Key Challenges in Small Sample Econometrics

    Use the interactive chart below to explore the main challenges posed by small samples in econometric analysis.
    """)

    challenges = {
        "Estimation Precision": "Wider confidence intervals and higher standard errors due to limited information",
        "Statistical Power": "Decreased ability to detect true effects, leading to false negatives",
        "Model Stability": "Parameter estimates highly sensitive to small changes in data or specification",
        "Non-normality": "Central Limit Theorem approximations break down, normal distribution less reliable",
        "Outlier Sensitivity": "Individual observations have greater influence on results",
        "Model Selection": "Increased risk of overfitting or selecting incorrect specifications",
        "Bias Magnitude": "Many estimators are only asymptotically unbiased; bias can be substantial in small samples",
        "Hypothesis Testing": "Test statistics often don't follow their asymptotic distributions",
        "Time Series Issues": "Special problems in dynamic models, unit root tests, and forecasting",
        "Measurement Error": "Errors-in-variables problems exacerbated in small samples"
    }

    selected_challenge = st.selectbox("Select a challenge to learn more:", list(challenges.keys()))

    st.info(challenges[selected_challenge])

    # Interactive chart showing the relationship between sample size and various statistical properties
    st.markdown("### How Statistical Properties Change with Sample Size")

    property_to_show = st.radio(
        "Select a statistical property to visualize:",
        ["Standard Error", "Power (for fixed effect size)", "Bias (for certain estimators)", "Distribution Shape"]
    )

    n_values = np.arange(5, 100)

    fig, ax = plt.subplots(figsize=(10, 6))

    if property_to_show == "Standard Error":
        # For standard error: SE = σ/√n
        sigma = 1
        se_values = sigma / np.sqrt(n_values)
        ax.plot(n_values, se_values)
        ax.set_ylabel("Standard Error")
        ax.set_title("Standard Error Decreases with Sample Size")
        ax.text(70, 0.3, "SE = σ/√n", fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

    elif property_to_show == "Power (for fixed effect size)":
        # For power: showing power for detecting a medium effect (Cohen's d = 0.5)
        d = 0.5  # medium effect size
        alpha = 0.05  # significance level
        # Non-centrality parameter
        ncp = d * np.sqrt(n_values)
        # Critical value for two-sided test
        crit = stats.norm.ppf(1 - alpha / 2)
        # Power
        power = 1 - stats.norm.cdf(crit - ncp)
        ax.plot(n_values, power)
        ax.axhline(y=0.8, color='r', linestyle='--', label="80% power")
        ax.set_ylabel("Power")
        ax.set_title("Statistical Power Increases with Sample Size")
        ax.legend()

    elif property_to_show == "Bias (for certain estimators)":
        # For AR(1) coefficient bias: approximately -1/T bias
        rho = 0.7  # true AR coefficient
        bias = -1 / n_values
        biased_estimate = rho + bias
        ax.plot(n_values, biased_estimate)
        ax.axhline(y=rho, color='r', linestyle='--', label="True value")
        ax.set_ylabel("Estimated Coefficient")
        ax.set_title("Bias in AR(1) Coefficient Estimate")
        ax.legend()

    elif property_to_show == "Distribution Shape":
        # For distribution shape: comparing t-distribution to normal
        sample_sizes = [5, 10, 30, 100]
        x = np.linspace(-4, 4, 1000)
        # Standard normal
        normal_pdf = stats.norm.pdf(x)
        ax.plot(x, normal_pdf, 'k-', label="Normal", alpha=0.7)

        # t-distributions with different degrees of freedom
        colors = ['r', 'g', 'b', 'c']
        for i, df in enumerate(sample_sizes):
            t_pdf = stats.t.pdf(x, df=df - 1)
            ax.plot(x, t_pdf, color=colors[i], label=f"t (df={df - 1})")

        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.set_title("Distribution Shape Approaches Normal as Sample Size Increases")
        ax.legend()

    ax.set_xlabel("Sample Size (n)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Offer a download link for the plot
    st.markdown(get_image_download_link(fig, f"{property_to_show.replace(' ', '_')}.png", "Download this figure"),
                unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ## Sample Size Determination

    Determining an adequate sample size depends on several factors. Use the calculator below to estimate required sample 
    sizes for different statistical analyses.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sample Size for Mean Estimation")

        conf_level = st.slider("Confidence Level (%)", 80, 99, 95, key="mean_conf_level")
        margin_error = st.slider("Margin of Error (% of std dev)", 5, 50, 20, key="mean_margin")

        z_value = stats.norm.ppf(1 - (1 - conf_level / 100) / 2)
        required_n = np.ceil((z_value / (margin_error / 100)) ** 2)

        st.success(f"Required sample size: **{int(required_n)}**")
        st.info(
            f"This ensures that your estimate will be within {margin_error}% of the true value with {conf_level}% confidence.")

    with col2:
        st.subheader("Sample Size for Hypothesis Testing")

        effect_size = st.select_slider(
            "Effect Size (Cohen's d)",
            options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            value=0.5,
            key="test_effect_size"
        )

        power_level = st.slider("Desired Power (%)", 70, 95, 80, key="test_power")
        alpha_level = st.slider("Significance Level (%)", 1, 10, 5, key="test_alpha")

        # Sample size calculation for two-sided test
        z_alpha = stats.norm.ppf(1 - alpha_level / 100 / 2)
        z_power = stats.norm.ppf(power_level / 100)

        required_n = np.ceil(((z_alpha + z_power) / effect_size) ** 2)

        st.success(f"Required sample size: **{int(required_n)}**")
        st.info(
            f"This gives you {power_level}% power to detect an effect size of {effect_size} at {alpha_level}% significance.")

###########################################
# PAGE: Visualization of Sample Size Effects
###########################################
elif selected_page == "Visualization of Sample Size Effects":
    st.title("Visualizing Small Sample Effects")

    st.markdown("""
    This section provides interactive visualizations to demonstrate how sample size affects various aspects of 
    econometric analysis. You can adjust parameters to see the effects in real-time.
    """)

    tab1, tab2, tab3 = st.tabs(["Distribution of Estimators", "Outlier Influence", "Sampling Variation"])

    with tab1:
        st.subheader("Distribution of Estimators by Sample Size")

        st.markdown("""
        This visualization shows how the sampling distribution of estimators changes with sample size. 
        With small samples, distributions are often wider and non-normal, affecting inference.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            true_beta = st.slider("True coefficient value", -2.0, 2.0, 1.0, 0.1)
            error_std = st.slider("Error standard deviation", 0.1, 5.0, 1.0, 0.1)

            sample_sizes = st.multiselect(
                "Sample sizes to compare",
                options=[5, 10, 20, 30, 50, 100, 200],
                default=[10, 30, 100]
            )

            n_simulations = st.slider(
                "Number of simulations",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100
            )

            show_normal = st.checkbox("Show normal approximation", value=True)

        with col2:
            if not sample_sizes:
                st.warning("Please select at least one sample size to visualize.")
            else:
                # Create simulations for each sample size
                fig, ax = plt.subplots(figsize=(10, 6))

                for n in sorted(sample_sizes):
                    betas = []

                    # Simulate OLS many times
                    for _ in range(n_simulations):
                        x = np.random.normal(0, 1, n)
                        e = np.random.normal(0, error_std, n)
                        y = true_beta * x + e

                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        betas.append(model.params[1])  # Get slope coefficient

                    # Plot the distribution of estimated betas
                    sns.kdeplot(betas, label=f"n = {n}", ax=ax)

                    # Add normal approximation if requested
                    if show_normal:
                        beta_mean = np.mean(betas)
                        beta_std = np.std(betas)
                        x_range = np.linspace(min(betas), max(betas), 1000)
                        y_range = stats.norm.pdf(x_range, beta_mean, beta_std)
                        ax.plot(x_range, y_range, '--', alpha=0.5)

                # Add vertical line for true value
                ax.axvline(x=true_beta, color='red', linestyle='--', alpha=0.7, label="True value")

                ax.set_xlabel("Estimated Coefficient")
                ax.set_ylabel("Density")
                ax.set_title("Sampling Distribution of Regression Coefficient by Sample Size")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Statistics table
                st.subheader("Sampling Distribution Statistics")

                stats_data = []
                for n in sorted(sample_sizes):
                    betas = []
                    for _ in range(n_simulations):
                        x = np.random.normal(0, 1, n)
                        e = np.random.normal(0, error_std, n)
                        y = true_beta * x + e

                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        betas.append(model.params[1])

                    stats_data.append({
                        "Sample Size": n,
                        "Mean": np.mean(betas),
                        "Bias": np.mean(betas) - true_beta,
                        "Std Dev": np.std(betas),
                        "95% CI Width": np.percentile(betas, 97.5) - np.percentile(betas, 2.5)
                    })

                st.table(pd.DataFrame(stats_data).set_index("Sample Size"))

    with tab2:
        st.subheader("Outlier Influence by Sample Size")

        st.markdown("""
        This demonstration shows how a single outlier can dramatically affect regression results, 
        with the effect being much more pronounced in small samples.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            base_sample_size = st.slider(
                "Base sample size",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )

            outlier_x = st.slider("Outlier X position", -10.0, 10.0, 5.0, 0.5)
            outlier_y = st.slider("Outlier Y position", -10.0, 10.0, 8.0, 0.5)

            show_outlier = st.checkbox("Include outlier", value=True)
            show_leverage = st.checkbox("Show leverage statistics", value=True)

            true_slope = 1.0

        with col2:
            # Generate base data
            np.random.seed(42)  # For reproducibility
            x_base = np.random.uniform(-3, 3, base_sample_size)
            y_base = true_slope * x_base + np.random.normal(0, 1, base_sample_size)

            # Create full dataset with and without outlier
            if show_outlier:
                x_full = np.append(x_base, outlier_x)
                y_full = np.append(y_base, outlier_y)
            else:
                x_full = x_base.copy()
                y_full = y_base.copy()

            # Fit models
            X_base = sm.add_constant(x_base)
            model_base = sm.OLS(y_base, X_base).fit()

            X_full = sm.add_constant(x_full)
            model_full = sm.OLS(y_full, X_full).fit()

            # Plot the results
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot data points
            ax.scatter(x_base, y_base, alpha=0.7, label="Base data")
            if show_outlier:
                ax.scatter([outlier_x], [outlier_y], color='red', s=100,
                           label="Outlier", zorder=10)

            # Plot regression lines
            x_line = np.linspace(min(x_full) - 1, max(x_full) + 1, 100)

            # Base model line
            y_base_pred = model_base.params[0] + model_base.params[1] * x_line
            ax.plot(x_line, y_base_pred, 'b-',
                    label=f"Without outlier: β = {model_base.params[1]:.3f}")

            # Full model line
            y_full_pred = model_full.params[0] + model_full.params[1] * x_line
            if show_outlier:
                ax.plot(x_line, y_full_pred, 'r-',
                        label=f"With outlier: β = {model_full.params[1]:.3f}")

            # True relationship
            ax.plot(x_line, true_slope * x_line, 'g--', alpha=0.7,
                    label=f"True relationship: β = {true_slope:.1f}")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(
                f"Effect of an Outlier in a Sample of {base_sample_size + (1 if show_outlier else 0)} Observations")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Display model statistics
            if show_outlier:
                st.subheader("Regression Statistics Comparison")

                stats_comparison = pd.DataFrame({
                    "Statistic": ["Slope Coefficient", "Standard Error", "t-statistic", "R-squared"],
                    "Without Outlier": [model_base.params[1], model_base.bse[1],
                                        model_base.tvalues[1], model_base.rsquared],
                    "With Outlier": [model_full.params[1], model_full.bse[1],
                                     model_full.tvalues[1], model_full.rsquared]
                }).set_index("Statistic")

                st.table(stats_comparison)

                if show_leverage:
                    # Calculate leverage and Cook's distance
                    leverage = model_full.get_influence().hat_matrix_diag
                    cooks_d = model_full.get_influence().cooks_distance[0]

                    # Highlight the outlier
                    lev_df = pd.DataFrame({
                        "Observation": range(1, len(x_full) + 1),
                        "X Value": x_full,
                        "Y Value": y_full,
                        "Leverage": leverage,
                        "Cook's Distance": cooks_d
                    })

                    st.subheader("Leverage Statistics")
                    st.write("Higher values indicate greater influence on the regression:")


                    # Highlight outlier row
                    def highlight_outlier(row):
                        if show_outlier and row.name == len(x_full) - 1:  # Outlier is last observation
                            return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                        return [''] * len(row)


                    st.dataframe(lev_df.style.apply(highlight_outlier, axis=1))

    with tab3:
        st.subheader("Sampling Variation Demonstration")

        st.markdown("""
        This interactive demo shows how sampling variation affects regression results. 
        With small samples, results can vary dramatically from one sample to another, even when drawn from the same population.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            sample_size = st.slider(
                "Sample size",
                min_value=5,
                max_value=200,
                value=15,
                key="sampling_variation_n"
            )

            noise_level = st.slider(
                "Noise level (standard deviation)",
                min_value=0.2,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="sampling_variation_noise"
            )

            n_samples = st.slider(
                "Number of random samples",
                min_value=1,
                max_value=10,
                value=5,
                key="sampling_variation_samples"
            )

            true_intercept = 2.0
            true_slope = 0.5

            # Explanation
            st.info("""
            This visualization randomly draws multiple samples from the same population
            and fits a regression line to each sample. With small samples, you'll notice
            much more variation in the regression lines.
            """)

            # New sample button
            if st.button("Generate New Samples"):
                st.session_state.random_seed = np.random.randint(1, 10000)
            else:
                if "random_seed" not in st.session_state:
                    st.session_state.random_seed = 42

        with col2:
            np.random.seed(st.session_state.random_seed)

            # Generate the full population
            x_population = np.linspace(-5, 5, 1000)
            y_population = true_intercept + true_slope * x_population

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the true relationship
            ax.plot(x_population, y_population, 'k--', alpha=0.7, label="True relationship")

            # Colors for different samples
            colors = plt.cm.tab10(np.linspace(0, 1, n_samples))

            # Generate and plot each sample
            sample_stats = []
            for i in range(n_samples):
                # Randomly sample x values
                x_sample = np.random.uniform(-5, 5, sample_size)
                # Generate corresponding y values with noise
                y_sample = true_intercept + true_slope * x_sample + np.random.normal(0, noise_level, sample_size)

                # Fit the model
                X_sample = sm.add_constant(x_sample)
                model = sm.OLS(y_sample, X_sample).fit()

                # Store statistics
                sample_stats.append({
                    "Sample": i + 1,
                    "Intercept": model.params[0],
                    "Slope": model.params[1],
                    "Std Error (Slope)": model.bse[1],
                    "t-value (Slope)": model.tvalues[1],
                    "p-value (Slope)": model.pvalues[1],
                    "Significant": model.pvalues[1] < 0.05
                })

                # Plot sample points
                ax.scatter(x_sample, y_sample, color=colors[i], alpha=0.5, s=30,
                           label=f"Sample {i + 1}")

                # Plot fitted line
                x_fit = np.linspace(-5, 5, 100)
                y_fit = model.params[0] + model.params[1] * x_fit
                ax.plot(x_fit, y_fit, color=colors[i], lw=2)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Regression Lines from {n_samples} Different Samples (n = {sample_size})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Display the statistics
            st.subheader("Sample Regression Statistics")
            stats_df = pd.DataFrame(sample_stats).set_index("Sample")

            # Format the p-values
            stats_df["p-value (Slope)"] = stats_df["p-value (Slope)"].apply(lambda p: f"{p:.4f}")

            st.dataframe(stats_df)

            # Summary message
            sig_count = stats_df["Significant"].sum()
            if sig_count < n_samples:
                st.warning(f"Only {sig_count} out of {n_samples} samples detected a statistically significant slope, "
                           f"even though the true slope is {true_slope}.")
            else:
                st.success(
                    f"All {n_samples} samples detected a statistically significant slope (true slope = {true_slope}).")

###########################################
# PAGE: Hypothesis Testing & Power
###########################################
elif selected_page == "Hypothesis Testing & Power":
    st.title("Hypothesis Testing & Statistical Power in Small Samples")

    st.markdown("""
    This section explores how small sample sizes affect hypothesis testing, particularly focusing on:

    1. Test statistics distributions
    2. Type I error rates (false positives)
    3. Statistical power (avoiding false negatives)
    4. Confidence interval width and interpretation

    Use the interactive tools below to explore these concepts.
    """)

    tab1, tab2, tab3 = st.tabs(["Test Statistic Distributions", "Power Analysis", "Multiple Testing"])

    with tab1:
        st.subheader("Test Statistic Distributions in Small Samples")

        st.markdown("""
        In large samples, test statistics like t and F approximately follow their theoretical distributions. 
        But in small samples, they can deviate significantly, affecting the reliability of p-values.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            test_stat = st.radio(
                "Test statistic to visualize:",
                ["t-statistic", "F-statistic", "Chi-square statistic"]
            )

            sample_sizes = st.multiselect(
                "Degrees of freedom to compare",
                options=[1, 2, 3, 5, 10, 20, 30, 50, 100],
                default=[5, 30, 100]
            )

            show_approx = st.checkbox("Show asymptotic approximation", value=True)

            st.markdown("""
            **Notes:**
            - For t-test, df = n-k (sample size minus parameters)
            - For F-test, shown with numerator df = 2
            - For Chi-square test, df is the number of restrictions
            """)

        with col2:
            if not sample_sizes:
                st.warning("Please select at least one sample size to visualize.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.linspace(-5, 5, 1000) if test_stat == "t-statistic" else np.linspace(0, 10, 1000)

                for df in sorted(sample_sizes):
                    if test_stat == "t-statistic":
                        y = stats.t.pdf(x, df=df)
                        label = f"t({df} df)"
                        if show_approx:
                            y_approx = stats.norm.pdf(x)
                            ax.plot(x, y_approx, 'k--', alpha=0.7, label="Normal (∞ df)")

                    elif test_stat == "F-statistic":
                        # For F-statistic, use 2 numerator df as an example
                        y = stats.f.pdf(x, dfn=2, dfd=df)
                        label = f"F(2,{df} df)"
                        if show_approx and df == max(sample_sizes):
                            y_approx = stats.chi2.pdf(x * 2, df=2) / 2
                            ax.plot(x, y_approx, 'k--', alpha=0.7, label="χ²/df (∞ df)")

                    else:  # Chi-square statistic
                        y = stats.chi2.pdf(x, df=df)
                        label = f"χ²({df} df)"

                    ax.plot(x, y, label=label)

                ax.set_xlabel("Value")
                ax.set_ylabel("Density")

                if test_stat == "t-statistic":
                    ax.set_title("t-distribution by Degrees of Freedom")
                    ax.set_xlim(-5, 5)
                elif test_stat == "F-statistic":
                    ax.set_title("F-distribution by Denominator Degrees of Freedom (numerator df = 2)")
                    ax.set_xlim(0, 5)
                else:
                    ax.set_title("Chi-square Distribution by Degrees of Freedom")
                    ax.set_xlim(0, max(10, max(sample_sizes) * 2))

                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Critical values comparison
                st.subheader("Critical Values Comparison")

                alpha_levels = [0.10, 0.05, 0.01]
                crit_data = []

                for df in sorted(sample_sizes):
                    row_data = {"Degrees of Freedom": df}

                    for alpha in alpha_levels:
                        if test_stat == "t-statistic":
                            # For two-sided test
                            crit_val = stats.t.ppf(1 - alpha / 2, df=df)
                            asymp_val = stats.norm.ppf(1 - alpha / 2)
                        elif test_stat == "F-statistic":
                            crit_val = stats.f.ppf(1 - alpha, dfn=2, dfd=df)
                            # Asymptotic value not easily comparable
                            asymp_val = np.nan
                        else:  # Chi-square
                            crit_val = stats.chi2.ppf(1 - alpha, df=df)
                            # Asymptotic value not applicable
                            asymp_val = np.nan

                        row_data[f"{alpha * 100}% level"] = f"{crit_val:.3f}"

                    crit_data.append(row_data)

                # Add asymptotic values for t-statistic
                if test_stat == "t-statistic" and show_approx:
                    asymp_row = {"Degrees of Freedom": "∞ (Normal)"}
                    for alpha in alpha_levels:
                        asymp_val = stats.norm.ppf(1 - alpha / 2)
                        asymp_row[f"{alpha * 100}% level"] = f"{asymp_val:.3f}"
                    crit_data.append(asymp_row)

                st.table(pd.DataFrame(crit_data).set_index("Degrees of Freedom"))

                if test_stat == "t-statistic":
                    st.info("""
                    **Practical Implication**: With small degrees of freedom, critical t-values are 
                    larger than the normal approximation (z-values). This means you need stronger 
                    evidence to reject the null hypothesis, accounting for the uncertainty from the small sample.
                    """)
                elif test_stat == "F-statistic":
                    st.info("""
                    **Practical Implication**: F-distributions with small denominator degrees of freedom 
                    have much larger critical values. This affects joint hypothesis tests and can lead to 
                    different conclusions compared to asymptotic Chi-square tests.
                    """)
                else:
                    st.info("""
                    **Practical Implication**: Chi-square distributions with small degrees of freedom
                    are highly skewed. In econometrics, this affects goodness-of-fit tests, likelihood ratio
                    tests, and specification tests in small samples.
                    """)

    with tab2:
        st.subheader("Statistical Power in Small Samples")

        st.markdown("""
        Statistical power is the probability of correctly rejecting a false null hypothesis.
        Small samples often lack sufficient power to detect meaningful effects, leading to false negatives.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            test_type = st.radio(
                "Type of test:",
                ["t-test (mean/coefficient)", "F-test (multiple restrictions)"]
            )

            effect_size_type = st.radio(
                "Effect size specification:",
                ["Standardized (Cohen's d)", "Raw units"]
            )

            if effect_size_type == "Standardized (Cohen's d)":
                effect_size = st.slider(
                    "Effect size (Cohen's d):",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.05
                )

                st.markdown("""
                **Effect Size Reference:**
                - 0.2: Small effect
                - 0.5: Medium effect
                - 0.8: Large effect
                """)
            else:
                mu0 = st.number_input("Null hypothesis value (μ₀)", value=0.0, step=0.1)
                mu1 = st.number_input("Alternative value (μ₁)", value=0.5, step=0.1)
                sigma = st.number_input("Standard deviation (σ)", value=1.0, step=0.1, min_value=0.1)

                effect_size = abs(mu1 - mu0) / sigma

            alpha_level = st.slider(
                "Significance level (α):",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01
            )

            if test_type == "F-test (multiple restrictions)":
                num_restrictions = st.slider(
                    "Number of restrictions (r):",
                    min_value=1,
                    max_value=10,
                    value=2
                )

                num_parameters = st.slider(
                    "Total number of parameters (k):",
                    min_value=num_restrictions,
                    max_value=20,
                    value=5
                )

        with col2:
            # Create power curve
            fig, ax = plt.subplots(figsize=(10, 6))

            # Range of sample sizes
            n_values = np.arange(5, 201)

            # Calculate power for each sample size
            power_values = []

            if test_type == "t-test (mean/coefficient)":
                for n in n_values:
                    # Non-centrality parameter
                    ncp = effect_size * np.sqrt(n)

                    # Critical value (two-sided test)
                    crit = stats.t.ppf(1 - alpha_level / 2, df=n - 1)

                    # Power calculation
                    power = 1 - stats.nct.cdf(crit, df=n - 1, nc=ncp)
                    power_values.append(power)
            else:
                # F-test power
                for n in n_values:
                    # Skip if not enough degrees of freedom
                    if n <= num_parameters:
                        power_values.append(np.nan)
                        continue

                    # Non-centrality parameter (approximation)
                    # Using r*f² as NCP where f² is effect size squared times n
                    ncp = num_restrictions * (effect_size ** 2) * n / num_parameters

                    # Critical value
                    crit = stats.f.ppf(1 - alpha_level, dfn=num_restrictions, dfd=n - num_parameters)

                    # Power calculation
                    power = 1 - stats.ncf.cdf(
                        crit,
                        dfn=num_restrictions,
                        dfd=n - num_parameters,
                        nc=ncp
                    )
                    power_values.append(power)

            # Plot the power curve
            ax.plot(n_values, power_values, 'b-', lw=2)

            # Add horizontal line at 80% power
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label="80% power")

            # Find minimum n for 80% power
            if max(power_values) >= 0.8:
                min_n_80 = n_values[np.where(np.array(power_values) >= 0.8)[0][0]]
                ax.axvline(x=min_n_80, color='g', linestyle='--', alpha=0.7,
                           label=f"n = {min_n_80} for 80% power")

            ax.set_xlabel("Sample Size (n)")
            ax.set_ylabel("Power (1-β)")

            if test_type == "t-test (mean/coefficient)":
                ax.set_title(f"Power Analysis for t-test with Effect Size = {effect_size:.2f}, α = {alpha_level}")
            else:
                ax.set_title(
                    f"Power Analysis for F-test with {num_restrictions} restrictions, Effect Size = {effect_size:.2f}, α = {alpha_level}")

            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()

            st.pyplot(fig)

            # Interactive power calculator
            st.subheader("Calculate Power for Specific Sample Size")

            specific_n = st.number_input(
                "Enter sample size:",
                min_value=5 if test_type == "t-test (mean/coefficient)" else num_parameters + 1,
                max_value=1000,
                value=30
            )

            if test_type == "t-test (mean/coefficient)":
                ncp = effect_size * np.sqrt(specific_n)
                crit = stats.t.ppf(1 - alpha_level / 2, df=specific_n - 1)
                specific_power = 1 - stats.nct.cdf(crit, df=specific_n - 1, nc=ncp)

                st.metric(
                    label=f"Power for n = {specific_n}",
                    value=f"{specific_power:.2%}"
                )

                # Explanation
                if specific_power < 0.8:
                    st.warning(f"""
                    With a sample size of {specific_n}, you have only {specific_power:.1%} power to detect 
                    an effect size of {effect_size:.2f}. This means you have a {(1 - specific_power):.1%} chance 
                    of failing to detect a true effect of this magnitude.
                    """)
                else:
                    st.success(f"""
                    With a sample size of {specific_n}, you have {specific_power:.1%} power to detect 
                    an effect size of {effect_size:.2f}.
                    """)
            else:
                if specific_n <= num_parameters:
                    st.error(f"Sample size must be greater than the number of parameters ({num_parameters}).")
                else:
                    ncp = num_restrictions * (effect_size ** 2) * specific_n / num_parameters
                    crit = stats.f.ppf(1 - alpha_level, dfn=num_restrictions, dfd=specific_n - num_parameters)
                    specific_power = 1 - stats.ncf.cdf(crit, dfn=num_restrictions, dfd=specific_n - num_parameters,
                                                       nc=ncp)

                    st.metric(
                        label=f"Power for n = {specific_n}",
                        value=f"{specific_power:.2%}"
                    )

                    # Explanation
                    if specific_power < 0.8:
                        st.warning(f"""
                        With a sample size of {specific_n}, you have only {specific_power:.1%} power to detect 
                        this effect. For F-tests with {num_restrictions} restrictions and {num_parameters} total parameters,
                        this increases the risk of Type II errors (false negatives).
                        """)
                    else:
                        st.success(f"""
                        With a sample size of {specific_n}, you have {specific_power:.1%} power to detect 
                        this effect in your F-test with {num_restrictions} restrictions.
                        """)

    with tab3:
        st.subheader("Multiple Testing in Small Samples")

        st.markdown("""
        When conducting multiple hypothesis tests, the probability of false positives increases.
        This is particularly problematic in small samples where individual tests already have limitations.
        """)

        st.info("""
        **Multiple Testing Problem**

        If you conduct 20 independent hypothesis tests at a 5% significance level, 
        the probability of at least one false positive is:

        P(at least one false positive) = 1 - (1 - 0.05)^20 ≈ 64%

        This demonstrates why correction methods are essential, especially with small samples.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            num_tests = st.slider(
                "Number of hypothesis tests:",
                min_value=1,
                max_value=100,
                value=20
            )

            alpha_individual = st.slider(
                "Individual test significance level (α):",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01
            )

            correction_method = st.radio(
                "Multiple testing correction method:",
                ["None", "Bonferroni", "Holm-Bonferroni", "Benjamini-Hochberg (FDR)"]
            )

            # Simulation parameters
            true_nulls_prop = st.slider(
                "Proportion of true nulls in simulation:",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05
            )

            effect_size_alt = st.slider(
                "Effect size for alternatives:",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1
            )

            sample_size_sim = st.slider(
                "Sample size for simulation:",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )

            st.markdown("""
            **Correction Methods:**
            - **Bonferroni**: α' = α/m (conservative)
            - **Holm-Bonferroni**: Sequential correction
            - **Benjamini-Hochberg**: Controls false discovery rate
            """)

        with col2:
            # Calculate familywise error rate
            fwer = 1 - (1 - alpha_individual) ** num_tests

            # Display FWER
            st.metric(
                label="Familywise Error Rate (no correction)",
                value=f"{fwer:.2%}",
                delta=f"{fwer - alpha_individual:.1%} higher than individual α" if num_tests > 1 else None,
                delta_color="inverse"
            )

            # Adjusted alpha values
            if correction_method == "Bonferroni":
                adjusted_alpha = alpha_individual / num_tests
                st.metric(
                    label="Bonferroni-adjusted significance level",
                    value=f"{adjusted_alpha:.4f}"
                )

            # Create simulation visualization
            st.subheader("Simulation of Multiple Testing")

            # Run simulation
            np.random.seed(42)

            # Number of true and false nulls
            num_true_nulls = int(num_tests * true_nulls_prop)
            num_false_nulls = num_tests - num_true_nulls

            # Generate p-values
            # For true nulls: p-values are uniform
            p_values_true = np.random.uniform(0, 1, num_true_nulls)

            # For false nulls: generate t-statistics with non-centrality parameter
            # and convert to p-values
            ncp = effect_size_alt * np.sqrt(sample_size_sim)
            t_stats_false = np.random.standard_t(df=sample_size_sim - 1, size=num_false_nulls) + ncp
            p_values_false = 2 * (1 - stats.t.cdf(np.abs(t_stats_false), df=sample_size_sim - 1))

            # Combine p-values
            p_values = np.concatenate([p_values_true, p_values_false])
            true_status = np.concatenate([np.ones(num_true_nulls), np.zeros(num_false_nulls)])

            # Create dataframe
            results_df = pd.DataFrame({
                "Test": [f"Test {i + 1}" for i in range(num_tests)],
                "Null Hypothesis": ["True" if s == 1 else "False" for s in true_status],
                "p-value": p_values
            })

            # Sort by p-value for visualization
            results_df = results_df.sort_values("p-value")

            # Apply corrections
            if correction_method == "Bonferroni":
                results_df["Adjusted p-value"] = np.minimum(results_df["p-value"] * num_tests, 1.0)
                results_df["Significant"] = results_df["Adjusted p-value"] < alpha_individual
            elif correction_method == "Holm-Bonferroni":
                # Holm-Bonferroni method
                results_df = results_df.reset_index(drop=True)
                adjusted_values = []
                for i in range(len(results_df)):
                    adjusted_values.append(min(1, results_df.loc[i, "p-value"] * (num_tests - i)))

                # Ensure monotonicity
                for i in range(len(adjusted_values) - 1, 0, -1):
                    adjusted_values[i - 1] = min(adjusted_values[i - 1], adjusted_values[i])

                results_df["Adjusted p-value"] = adjusted_values
                results_df["Significant"] = results_df["Adjusted p-value"] < alpha_individual
            elif correction_method == "Benjamini-Hochberg (FDR)":
                # Benjamini-Hochberg method
                results_df = results_df.reset_index(drop=True)
                adjusted_values = []
                for i in range(len(results_df)):
                    adjusted_values.append(results_df.loc[i, "p-value"] * num_tests / (i + 1))

                # Ensure monotonicity
                for i in range(len(adjusted_values) - 1, 0, -1):
                    adjusted_values[i - 1] = min(adjusted_values[i - 1], adjusted_values[i])

                results_df["Adjusted p-value"] = adjusted_values
                results_df["Significant"] = results_df["Adjusted p-value"] < alpha_individual
            else:
                # No correction
                results_df["Adjusted p-value"] = results_df["p-value"]
                results_df["Significant"] = results_df["p-value"] < alpha_individual

            # Count results
            true_pos = sum((results_df["Null Hypothesis"] == "False") & results_df["Significant"])
            false_pos = sum((results_df["Null Hypothesis"] == "True") & results_df["Significant"])
            true_neg = sum((results_df["Null Hypothesis"] == "True") & ~results_df["Significant"])
            false_neg = sum((results_df["Null Hypothesis"] == "False") & ~results_df["Significant"])

            # Plot the results
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create horizontal line for alpha
            ax.axhline(alpha_individual, color='r', linestyle='--',
                       label=f"α = {alpha_individual}")

            # Plot points
            scatter = ax.scatter(
                range(len(results_df)),
                results_df["p-value" if correction_method == "None" else "Adjusted p-value"],
                c=results_df["Null Hypothesis"].map({"True": "blue", "False": "red"}),
                alpha=0.7,
                s=50
            )

            ax.set_xlabel("Tests (ordered by p-value)")
            ax.set_ylabel("p-value" if correction_method == "None" else "Adjusted p-value")
            title = "Multiple Testing Simulation"
            if correction_method != "None":
                title += f" with {correction_method} Correction"
            ax.set_title(title)

            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='True Null'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False Null')
            ]
            ax.legend(handles=legend_elements)

            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, min(1.05, results_df["Adjusted p-value"].max() * 1.1))

            st.pyplot(fig)

            # Show confusion matrix
            st.subheader("Testing Results")

            confusion_matrix = pd.DataFrame({
                "": ["H₀ True", "H₀ False"],
                "Rejected H₀": [false_pos, true_pos],
                "Failed to Reject H₀": [true_neg, false_neg]
            }).set_index("")

            st.table(confusion_matrix)

            # Calculate performance metrics
            fpr = false_pos / num_true_nulls if num_true_nulls > 0 else 0
            fnr = false_neg / num_false_nulls if num_false_nulls > 0 else 0
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("False Positive Rate", f"{fpr:.2%}")
            with col2:
                st.metric("False Negative Rate", f"{fnr:.2%}")
            with col3:
                st.metric("Precision", f"{precision:.2%}")

            # Provide interpretation
            if correction_method == "None":
                st.warning("""
                **Without correction**, the false positive rate is high when running multiple tests.
                This is especially problematic in small samples where individual tests already have limitations.
                """)
            elif false_pos == 0:
                st.success(f"""
                The {correction_method} correction successfully controlled false positives.
                However, this comes at the cost of {false_neg} false negatives (missed true effects).
                """)
            else:
                st.info(f"""
                The {correction_method} correction reduced false positives compared to no correction,
                but still allowed {false_pos} false discoveries.
                """)

            # Show top 10 results in detail
            st.subheader("Detailed Test Results (Top 10)")

            display_df = results_df.head(10).copy()
            display_df["p-value"] = display_df["p-value"].apply(lambda p: f"{p:.4f}")
            display_df["Adjusted p-value"] = display_df["Adjusted p-value"].apply(lambda p: f"{p:.4f}")


            def highlight_significant(row):
                return ['background-color: rgba(0, 255, 0, 0.2)' if row["Significant"] else '' for _ in row]


            st.dataframe(display_df.style.apply(highlight_significant, axis=1))

###########################################
# PAGE: Coefficient Estimation & Bias
###########################################
elif selected_page == "Coefficient Estimation & Bias":
    st.title("Coefficient Estimation & Bias in Small Samples")

    st.markdown("""
    This section explores how small sample sizes affect coefficient estimation, focusing on:

    1. Bias in various econometric estimators
    2. Estimation precision and confidence intervals
    3. Sensitivity to specification changes
    4. Methods to improve estimation in small samples
    """)

    tab1, tab2, tab3 = st.tabs(["Estimator Bias", "Instrumental Variables", "Bias Reduction Methods"])

    with tab1:
        st.subheader("Small Sample Bias in Econometric Estimators")

        st.markdown("""
        Many econometric estimators are only asymptotically unbiased, meaning they exhibit bias in small samples.
        This tool visualizes how this bias changes with sample size for different estimators.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            estimator_type = st.radio(
                "Estimator to analyze:",
                ["OLS with exogeneity", "OLS with omitted variable", "Dynamic model (AR)", "Maximum Likelihood"]
            )

            # Common parameters for all estimators
            n_simulations = st.slider(
                "Number of simulations",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="bias_simulations"
            )

            show_dist = st.checkbox("Show distribution of estimates", value=True)

            # Model-specific parameters
            if estimator_type == "OLS with exogeneity":
                true_beta = st.slider("True coefficient value", -2.0, 2.0, 1.0, 0.1, key="ols_true_beta")
                error_std = st.slider("Error standard deviation", 0.1, 5.0, 1.0, 0.1, key="ols_error_std")

            elif estimator_type == "OLS with omitted variable":
                beta_1 = st.slider("True coefficient (β₁)", -2.0, 2.0, 1.0, 0.1, key="omit_beta1")
                beta_2 = st.slider("Omitted variable coefficient (β₂)", -2.0, 2.0, 0.5, 0.1, key="omit_beta2")
                corr_x1x2 = st.slider("Correlation between x₁ and x₂", -0.9, 0.9, 0.5, 0.1, key="omit_corr")

            elif estimator_type == "Dynamic model (AR)":
                ar_coef = st.slider("True AR coefficient (ρ)", -0.9, 0.9, 0.7, 0.1, key="ar_coef")

            elif estimator_type == "Maximum Likelihood":
                dist_family = st.radio(
                    "Distribution family",
                    ["Logit", "Poisson"]
                )
                true_beta_ml = st.slider("True coefficient value", -2.0, 2.0, 1.0, 0.1, key="ml_true_beta")

        with col2:
            # Range of sample sizes to analyze
            sample_sizes = [10, 20, 30, 50, 100, 200, 500]

            # Store results for each sample size
            bias_results = []
            estimator_distributions = {}

            # Run simulations for each sample size
            for n in sample_sizes:
                estimates = []

                for _ in range(n_simulations):
                    if estimator_type == "OLS with exogeneity":
                        # Simple OLS model with exogeneity
                        x = np.random.normal(0, 1, n)
                        e = np.random.normal(0, error_std, n)
                        y = true_beta * x + e

                        # Estimate model
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        estimates.append(model.params[1])  # Extract slope parameter

                    elif estimator_type == "OLS with omitted variable":
                        # Generate correlated predictors
                        x1 = np.random.normal(0, 1, n)
                        x2 = corr_x1x2 * x1 + np.sqrt(1 - corr_x1x2 ** 2) * np.random.normal(0, 1, n)

                        # Generate outcome
                        e = np.random.normal(0, 1, n)
                        y = beta_1 * x1 + beta_2 * x2 + e

                        # Estimate model with x1 only (omitting x2)
                        X = sm.add_constant(x1)
                        model = sm.OLS(y, X).fit()
                        estimates.append(model.params[1])  # Extract slope parameter

                    elif estimator_type == "Dynamic model (AR)":
                        # Generate AR(1) process
                        y = np.zeros(n)
                        e = np.random.normal(0, 1, n)

                        for t in range(1, n):
                            y[t] = ar_coef * y[t - 1] + e[t]

                        # Estimate AR coefficient
                        y_lag = y[:-1]
                        y_current = y[1:]

                        # Simple regression without constant
                        model = sm.OLS(y_current, y_lag).fit()
                        estimates.append(model.params[0])

                    elif estimator_type == "Maximum Likelihood":
                        if dist_family == "Logit":
                            # Generate logit data
                            x = np.random.normal(0, 1, n)
                            p = 1 / (1 + np.exp(-true_beta_ml * x))
                            y = np.random.binomial(1, p, n)

                            # Estimate logit model
                            X = sm.add_constant(x)
                            try:
                                model = sm.Logit(y, X).fit(disp=0)
                                estimates.append(model.params[1])
                            except:
                                # If convergence fails, skip this iteration
                                continue

                        elif dist_family == "Poisson":
                            # Generate Poisson data
                            x = np.random.normal(0, 1, n)
                            lambda_val = np.exp(true_beta_ml * x)
                            y = np.random.poisson(lambda_val, n)

                            # Estimate Poisson model
                            X = sm.add_constant(x)
                            try:
                                model = sm.Poisson(y, X).fit(disp=0)
                                estimates.append(model.params[1])
                            except:
                                # If convergence fails, skip this iteration
                                continue

                # Store distribution for this sample size
                estimator_distributions[n] = estimates

                # Calculate summary statistics
                mean_est = np.mean(estimates)

                if estimator_type == "OLS with exogeneity":
                    true_value = true_beta
                elif estimator_type == "OLS with omitted variable":
                    true_value = beta_1 + beta_2 * corr_x1x2  # Omitted variable bias formula
                elif estimator_type == "Dynamic model (AR)":
                    true_value = ar_coef
                else:  # Maximum Likelihood
                    true_value = true_beta_ml

                bias = mean_est - true_value
                rmse = np.sqrt(np.mean((np.array(estimates) - true_value) ** 2))

                bias_results.append({
                    "Sample Size": n,
                    "Mean Estimate": mean_est,
                    "Bias": bias,
                    "RMSE": rmse,
                    "Standard Error": np.std(estimates)
                })

            # Create bias plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract data for plotting
            n_values = [result["Sample Size"] for result in bias_results]
            bias_values = [result["Bias"] for result in bias_results]

            # Plot bias
            ax.plot(n_values, bias_values, 'b-o', linewidth=2)

            # Add horizontal line at zero (no bias)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)

            ax.set_xlabel("Sample Size (n)")
            ax.set_ylabel("Bias")

            if estimator_type == "OLS with exogeneity":
                ax.set_title(f"Bias in OLS Estimator (True β = {true_beta})")
            elif estimator_type == "OLS with omitted variable":
                ax.set_title(f"Bias in OLS with Omitted Variable (β₁ = {beta_1}, β₂ = {beta_2}, corr = {corr_x1x2})")
            elif estimator_type == "Dynamic model (AR)":
                ax.set_title(f"Bias in AR(1) Coefficient Estimator (True ρ = {ar_coef})")
            else:
                ax.set_title(f"Bias in {dist_family} ML Estimator (True β = {true_beta_ml})")

            ax.grid(True, alpha=0.3)

            # Set x-axis to log scale
            ax.set_xscale('log')
            ax.set_xticks(n_values)
            ax.set_xticklabels([str(n) for n in n_values])

            st.pyplot(fig)

            # Display a table of results
            st.subheader("Bias and Precision by Sample Size")

            results_df = pd.DataFrame(bias_results).set_index("Sample Size")

            # Format the data for display
            display_df = results_df.copy()
            for col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

            st.table(display_df)

            # Show distribution plot if requested
            if show_dist:
                st.subheader("Distribution of Estimates by Sample Size")

                fig, ax = plt.subplots(figsize=(10, 6))

                # Choose a subset of sample sizes to show
                sizes_to_show = [10, 30, 100, 500] if 500 in sample_sizes else [10, 30, 100, 200]
                sizes_to_show = [s for s in sizes_to_show if s in sample_sizes]

                for n in sizes_to_show:
                    sns.kdeplot(estimator_distributions[n], label=f"n = {n}", ax=ax)

                # Add vertical line for true value
                if estimator_type == "OLS with exogeneity":
                    ax.axvline(x=true_beta, color='r', linestyle='--', alpha=0.7, label="True value")
                elif estimator_type == "OLS with omitted variable":
                    true_val = beta_1 + beta_2 * corr_x1x2
                    ax.axvline(x=true_val, color='r', linestyle='--', alpha=0.7,
                               label=f"Probability limit ({true_val:.2f})")
                    ax.axvline(x=beta_1, color='g', linestyle='--', alpha=0.7,
                               label=f"True β₁ ({beta_1})")
                elif estimator_type == "Dynamic model (AR)":
                    ax.axvline(x=ar_coef, color='r', linestyle='--', alpha=0.7, label=f"True ρ ({ar_coef})")
                else:
                    ax.axvline(x=true_beta_ml, color='r', linestyle='--', alpha=0.7, label=f"True β ({true_beta_ml})")

                ax.set_xlabel("Estimated Coefficient")
                ax.set_ylabel("Density")
                ax.set_title("Distribution of Coefficient Estimates by Sample Size")
                ax.grid(True, alpha=0.3)
                ax.legend()

                st.pyplot(fig)

            # Add explanation based on the estimator type
            if estimator_type == "OLS with exogeneity":
                st.info("""
                **Interpretation:** With a correctly specified model and exogenous regressors, OLS is unbiased even in 
                small samples. Any deviation from zero bias in the chart is due to simulation randomness. 

                Note that while the estimator is unbiased, the precision (standard error) improves with sample size.
                """)
            elif estimator_type == "OLS with omitted variable":
                st.warning(f"""
                **Interpretation:** When an important variable (x₂) is omitted, OLS suffers from omitted variable bias.
                The bias equals β₂ × correlation(x₁,x₂) = {beta_2} × {corr_x1x2} = {beta_2 * corr_x1x2:.4f}.

                This bias persists regardless of sample size, demonstrating that large samples don't solve specification errors.
                """)
            elif estimator_type == "Dynamic model (AR)":
                st.warning(f"""
                **Interpretation:** The OLS estimator of the AR(1) coefficient has a downward bias of approximately -1/T in small samples.
                This is known as the Hurwicz bias or the Nickell bias in panel data contexts.

                Notice how the bias decreases as sample size increases, approaching zero in large samples.
                """)
            else:
                if dist_family == "Logit":
                    st.warning("""
                    **Interpretation:** Maximum Likelihood estimators like Logit have bias of order O(1/n) in small samples.

                    Small samples in Logit models can also lead to separation problems where certain coefficient combinations
                    perfectly predict outcomes, leading to infinite parameter estimates.
                    """)
                else:  # Poisson
                    st.warning("""
                    **Interpretation:** Poisson ML estimators exhibit small-sample bias, especially with few observations per 
                    parameter. The bias reduces at rate 1/n as sample size increases.

                    In very small samples, the Poisson model may also suffer from convergence issues.
                    """)

    with tab2:
        st.subheader("Instrumental Variables in Small Samples")

        st.markdown("""
        Instrumental Variables (IV) estimation is particularly sensitive to small sample issues,
        especially when instruments are weak. This tool demonstrates IV behavior in small samples.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            # Model parameters
            true_beta = st.slider("True coefficient (β)", -2.0, 2.0, 1.0, 0.1, key="iv_true_beta")

            # Instrument strength
            instrument_strength = st.select_slider(
                "Instrument strength (first-stage F)",
                options=["Very weak (F≈2)", "Weak (F≈5)", "Moderate (F≈10)", "Strong (F≈20)", "Very strong (F≈50)"]
            )

            # Convert text to numeric strength
            strength_map = {
                "Very weak (F≈2)": 0.2,
                "Weak (F≈5)": 0.4,
                "Moderate (F≈10)": 0.6,
                "Strong (F≈20)": 0.8,
                "Very strong (F≈50)": 0.95
            }
            instrument_corr = strength_map[instrument_strength]

            # Endogeneity
            endogeneity = st.slider(
                "Endogeneity strength (corr(x,e))",
                min_value=0.0,
                max_value=0.9,
                value=0.5,
                step=0.1,
                key="iv_endogeneity"
            )

            # Simulations
            n_simulations = st.slider(
                "Number of simulations",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="iv_simulations"
            )

            compare_estimators = st.checkbox("Compare OLS and 2SLS", value=True)

        with col2:
            # Range of sample sizes to analyze
            sample_sizes = [10, 20, 30, 50, 100, 200, 500]

            # Store results for each sample size
            iv_results = []
            ols_results = []  # For comparison

            # Run simulations for each sample size
            for n in sample_sizes:
                iv_estimates = []
                ols_estimates = []
                first_stage_f = []

                for _ in range(n_simulations):
                    # Generate instrument
                    z = np.random.normal(0, 1, n)

                    # Generate error terms with correlation
                    e_x = np.random.normal(0, 1, n)
                    e_y = endogeneity * e_x + np.sqrt(1 - endogeneity ** 2) * np.random.normal(0, 1, n)

                    # Generate endogenous variable with instrument correlation
                    x = instrument_corr * z + np.sqrt(1 - instrument_corr ** 2) * e_x

                    # Generate outcome
                    y = true_beta * x + e_y

                    # Estimate OLS model (biased due to endogeneity)
                    X = sm.add_constant(x)
                    ols_model = sm.OLS(y, X).fit()
                    ols_estimates.append(ols_model.params[1])

                    # First stage regression
                    Z = sm.add_constant(z)
                    first_stage = sm.OLS(x, Z).fit()

                    # Calculate first-stage F-statistic
                    f_stat = first_stage.tvalues[1] ** 2
                    first_stage_f.append(f_stat)

                    # Second stage
                    x_hat = first_stage.predict(Z)
                    X_hat = sm.add_constant(x_hat)

                    # 2SLS
                    try:
                        iv_model = sm.OLS(y, X_hat).fit()
                        iv_estimates.append(iv_model.params[1])
                    except:
                        # Skip if estimation fails
                        continue

                # Calculate summary statistics for IV
                mean_iv = np.mean(iv_estimates)
                bias_iv = mean_iv - true_beta
                rmse_iv = np.sqrt(np.mean((np.array(iv_estimates) - true_beta) ** 2))

                iv_results.append({
                    "Sample Size": n,
                    "Mean Estimate": mean_iv,
                    "Bias": bias_iv,
                    "RMSE": rmse_iv,
                    "Standard Error": np.std(iv_estimates),
                    "Mean First-Stage F": np.mean(first_stage_f)
                })

                # Calculate summary statistics for OLS
                mean_ols = np.mean(ols_estimates)
                bias_ols = mean_ols - true_beta

                ols_results.append({
                    "Sample Size": n,
                    "Mean Estimate": mean_ols,
                    "Bias": bias_ols
                })

            # Create bias plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract data for plotting
            n_values = [result["Sample Size"] for result in iv_results]
            bias_iv_values = [result["Bias"] for result in iv_results]

            # Plot IV bias
            ax.plot(n_values, bias_iv_values, 'b-o', linewidth=2, label="2SLS Bias")

            # Add OLS bias for comparison if requested
            if compare_estimators:
                bias_ols_values = [result["Bias"] for result in ols_results]
                ax.plot(n_values, bias_ols_values, 'r-^', linewidth=2, label="OLS Bias")

            # Add horizontal line at zero (no bias)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)

            ax.set_xlabel("Sample Size (n)")
            ax.set_ylabel("Bias")
            ax.set_title(f"IV Bias by Sample Size (β = {true_beta}, Instrument Strength: {instrument_strength})")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Set x-axis to log scale
            ax.set_xscale('log')
            ax.set_xticks(n_values)
            ax.set_xticklabels([str(n) for n in n_values])

            st.pyplot(fig)

            # Create RMSE comparison plot
            if compare_estimators:
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                # Extract RMSE data
                rmse_iv_values = [result["RMSE"] for result in iv_results]
                rmse_ols_values = [np.sqrt(np.mean((np.array(ols_estimates) - true_beta) ** 2))
                                   for ols_estimates in [
                                       [ols_results[i]["Mean Estimate"]] for i in range(len(ols_results))
                                   ]]

                # Plot RMSE
                ax2.plot(n_values, rmse_iv_values, 'b-o', linewidth=2, label="2SLS RMSE")
                ax2.plot(n_values, rmse_ols_values, 'r-^', linewidth=2, label="OLS RMSE")

                ax2.set_xlabel("Sample Size (n)")
                ax2.set_ylabel("Root Mean Squared Error")
                ax2.set_title("Efficiency Comparison: 2SLS vs OLS")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                # Set x-axis to log scale
                ax2.set_xscale('log')
                ax2.set_xticks(n_values)
                ax2.set_xticklabels([str(n) for n in n_values])

                st.pyplot(fig2)

            # Display results table
            st.subheader("2SLS Estimation Results by Sample Size")

            results_df = pd.DataFrame(iv_results).set_index("Sample Size")

            # Format for display
            display_df = results_df.copy()
            for col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

            st.table(display_df)

            # Distribution of IV estimator in small samples
            st.subheader("Distribution of 2SLS Estimates")

            # Choose sample sizes to show
            small_n = 20
            large_n = 200 if 200 in sample_sizes else 100

            # Find indices
            small_idx = sample_sizes.index(small_n)
            large_idx = sample_sizes.index(large_n)

            # Extract simulation data for these sample sizes
            iv_small = [est for est in iv_estimates if -10 < est < 10]  # Filter extreme values

            # Generate new data for the large sample
            large_iv_estimates = []

            for _ in range(n_simulations):
                z = np.random.normal(0, 1, large_n)
                e_x = np.random.normal(0, 1, large_n)
                e_y = endogeneity * e_x + np.sqrt(1 - endogeneity ** 2) * np.random.normal(0, 1, large_n)
                x = instrument_corr * z + np.sqrt(1 - instrument_corr ** 2) * e_x
                y = true_beta * x + e_y

                # First stage
                Z = sm.add_constant(z)
                first_stage = sm.OLS(x, Z).fit()
                x_hat = first_stage.predict(Z)
                X_hat = sm.add_constant(x_hat)

                # 2SLS
                try:
                    iv_model = sm.OLS(y, X_hat).fit()
                    large_iv_estimates.append(iv_model.params[1])
                except:
                    continue

            iv_large = [est for est in large_iv_estimates if -10 < est < 10]  # Filter extreme values

            fig3, ax3 = plt.subplots(figsize=(10, 6))

            sns.kdeplot(iv_small, label=f"n = {small_n}", ax=ax3)
            sns.kdeplot(iv_large, label=f"n = {large_n}", ax=ax3)

            # Add vertical line for true value
            ax3.axvline(x=true_beta, color='r', linestyle='--', alpha=0.7, label=f"True β ({true_beta})")

            if compare_estimators:
                # Find OLS plim for large sample
                ols_plim = true_beta + endogeneity  # Simplified for this demonstration
                ax3.axvline(x=ols_plim, color='g', linestyle='--', alpha=0.7,
                            label=f"OLS probability limit ({ols_plim:.2f})")

            ax3.set_xlabel("Estimated Coefficient")
            ax3.set_ylabel("Density")
            ax3.set_title("Distribution of 2SLS Estimates by Sample Size")
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # Set reasonable x-axis limits
            q_low = np.percentile(iv_small + iv_large, 1)
            q_high = np.percentile(iv_small + iv_large, 99)
            ax3.set_xlim(max(q_low, -5), min(q_high, 5))

            st.pyplot(fig3)

            # Add explanation
            st.info("""
            **Key Findings on IV/2SLS in Small Samples:**

            1. **Bias toward OLS:** In small samples, 2SLS is biased toward OLS, with the bias decreasing as sample size increases

            2. **High Variance:** 2SLS has much higher variance than OLS, especially with weak instruments

            3. **Non-normal Sampling Distribution:** With weak instruments, the distribution can be non-normal and heavy-tailed

            4. **Bias-Variance Tradeoff:** In terms of MSE, OLS might outperform 2SLS in small samples despite bias

            5. **First-stage F-statistic:** Critical for assessing instrument strength (rule of thumb: F > 10)
            """)

            # Show a more detailed explanation based on instrument strength
            if instrument_strength in ["Very weak (F≈2)", "Weak (F≈5)"]:
                st.warning("""
                **Weak Instrument Warning:** The simulation uses weak instruments which cause serious problems in small samples:

                - Substantial bias toward OLS
                - Extremely high variance
                - Confidence intervals that fail to include the true parameter value at the nominal rate
                - Test statistics that don't follow standard distributions

                With such weak instruments, alternatives like LIML or robust confidence intervals may be preferable.
                """)

    with tab3:
        st.subheader("Bias Reduction Methods for Small Samples")

        st.markdown("""
        Several methods exist to reduce bias in small sample estimation. This tool demonstrates 
        the effectiveness of different bias correction techniques compared to standard estimators.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            model_type = st.radio(
                "Model type:",
                ["Dynamic model (AR)", "Binary choice (Logit/Probit)", "Panel data"]
            )

            # Model-specific parameters
            if model_type == "Dynamic model (AR)":
                true_rho = st.slider(
                    "True AR coefficient (ρ)",
                    min_value=-0.9,
                    max_value=0.9,
                    value=0.7,
                    step=0.1,
                    key="ar_true_rho"
                )

                correction_methods = st.multiselect(
                    "Bias correction methods:",
                    ["Standard OLS", "Analytical correction", "Bootstrap correction", "Indirect inference"],
                    default=["Standard OLS", "Analytical correction"]
                )

            elif model_type == "Binary choice (Logit/Probit)":
                model_variant = st.radio(
                    "Model variant:",
                    ["Logit", "Probit"]
                )

                true_beta_binary = st.slider(
                    "True coefficient (β)",
                    min_value=-2.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key="binary_true_beta"
                )

                correction_methods = st.multiselect(
                    "Bias correction methods:",
                    ["Standard ML", "Firth's penalized ML", "Jackknife correction"],
                    default=["Standard ML", "Firth's penalized ML"]
                )

            else:  # Panel data
                true_beta_panel = st.slider(
                    "True coefficient (β)",
                    min_value=-2.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key="panel_true_beta"
                )

                panel_type = st.radio(
                    "Panel model:",
                    ["Static panel", "Dynamic panel"]
                )

                if panel_type == "Dynamic panel":
                    true_lambda = st.slider(
                        "True lagged dependent coefficient (λ)",
                        min_value=-0.9,
                        max_value=0.9,
                        value=0.5,
                        step=0.1,
                        key="panel_true_lambda"
                    )

                correction_methods = st.multiselect(
                    "Estimation methods:",
                    ["Pooled OLS", "Fixed Effects", "Arellano-Bond", "Blundell-Bond"],
                    default=["Pooled OLS", "Fixed Effects"]
                )

            # Simulation parameters
            n_simulations = st.slider(
                "Number of simulations",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                key="correction_simulations"
            )

        with col2:
            # Determine sample sizes based on model type
            if model_type == "Dynamic model (AR)":
                sample_sizes = [10, 20, 30, 50, 100]
            elif model_type == "Binary choice (Logit/Probit)":
                sample_sizes = [20, 50, 100, 200]
            else:  # Panel data
                # Use (N, T) pairs where N*T is reasonable for simulation
                if panel_type == "Static panel":
                    sample_sizes = [(10, 3), (20, 5), (50, 5), (100, 10)]
                else:  # Dynamic panel
                    sample_sizes = [(10, 5), (20, 5), (50, 10), (100, 10)]

            # Prepare to store results
            if model_type != "Panel data":
                results = {method: [] for method in correction_methods}
            else:
                results = {method: [] for method in correction_methods}

            # Run simulations
            if model_type == "Dynamic model (AR)":
                for n in sample_sizes:
                    method_results = {method: [] for method in correction_methods}

                    for _ in range(n_simulations):
                        # Generate AR(1) process
                        y = np.zeros(n)
                        e = np.random.normal(0, 1, n)

                        for t in range(1, n):
                            y[t] = true_rho * y[t - 1] + e[t]

                        # Different estimation methods
                        y_lag = y[:-1]
                        y_current = y[1:]

                        for method in correction_methods:
                            if method == "Standard OLS":
                                # Simple regression without constant
                                model = sm.OLS(y_current, y_lag).fit()
                                estimate = model.params[0]

                            elif method == "Analytical correction":
                                # Kendall's analytical correction
                                model = sm.OLS(y_current, y_lag).fit()
                                raw_estimate = model.params[0]

                                # Apply correction formula: β' = β + (1+3β)/T
                                correction = (1 + 3 * raw_estimate) / (n - 1)
                                estimate = raw_estimate + correction

                            elif method == "Bootstrap correction":
                                # Simple regression first
                                model = sm.OLS(y_current, y_lag).fit()
                                raw_estimate = model.params[0]

                                # Bootstrap correction
                                bootstrap_estimates = []

                                # Limited number of bootstrap samples for speed
                                for _ in range(min(50, n_simulations)):
                                    # Generate bootstrap sample from fitted model
                                    y_boot = np.zeros(n)
                                    e_boot = np.random.normal(0, 1, n)

                                    for t in range(1, n):
                                        y_boot[t] = raw_estimate * y_boot[t - 1] + e_boot[t]

                                    # Estimate on bootstrap sample
                                    y_lag_boot = y_boot[:-1]
                                    y_current_boot = y_boot[1:]

                                    model_boot = sm.OLS(y_current_boot, y_lag_boot).fit()
                                    bootstrap_estimates.append(model_boot.params[0])

                                # Calculate bootstrap bias
                                bootstrap_bias = np.mean(bootstrap_estimates) - raw_estimate

                                # Apply correction
                                estimate = raw_estimate - bootstrap_bias

                            elif method == "Indirect inference":
                                # Simplified indirect inference (computationally intensive)
                                model = sm.OLS(y_current, y_lag).fit()
                                raw_estimate = model.params[0]

                                # Define grid of possible values around raw estimate
                                grid_values = np.linspace(max(-0.99, raw_estimate - 0.3),
                                                          min(0.99, raw_estimate + 0.3),
                                                          7)

                                # For each grid value, simulate data and calculate distance
                                distances = []

                                for grid_val in grid_values:
                                    simulated_estimates = []

                                    # Limited number of simulations for speed
                                    for _ in range(10):
                                        y_sim = np.zeros(n)
                                        e_sim = np.random.normal(0, 1, n)

                                        for t in range(1, n):
                                            y_sim[t] = grid_val * y_sim[t - 1] + e_sim[t]

                                        # Estimate on simulated data
                                        y_lag_sim = y_sim[:-1]
                                        y_current_sim = y_sim[1:]

                                        model_sim = sm.OLS(y_current_sim, y_lag_sim).fit()
                                        simulated_estimates.append(model_sim.params[0])

                                    # Calculate distance to raw estimate
                                    distance = abs(np.mean(simulated_estimates) - raw_estimate)
                                    distances.append(distance)

                                # Choose grid value with minimum distance
                                estimate = grid_values[np.argmin(distances)]

                            method_results[method].append(estimate)

                    # Calculate summary statistics
                    for method in correction_methods:
                        estimates = method_results[method]
                        mean_est = np.mean(estimates)
                        bias = mean_est - true_rho
                        rmse = np.sqrt(np.mean((np.array(estimates) - true_rho) ** 2))

                        results[method].append({
                            "Sample Size": n,
                            "Mean Estimate": mean_est,
                            "Bias": bias,
                            "RMSE": rmse,
                            "MAE": np.mean(np.abs(np.array(estimates) - true_rho))
                        })

                # Create comparison plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Plot bias
                for method in correction_methods:
                    bias_values = [result["Bias"] for result in results[method]]
                    ax1.plot(sample_sizes, bias_values, 'o-', label=method)

                ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7)
                ax1.set_xlabel("Sample Size (T)")
                ax1.set_ylabel("Bias")
                ax1.set_title(f"Bias Comparison for AR(1), ρ = {true_rho}")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot RMSE
                for method in correction_methods:
                    rmse_values = [result["RMSE"] for result in results[method]]
                    ax2.plot(sample_sizes, rmse_values, 'o-', label=method)

                ax2.set_xlabel("Sample Size (T)")
                ax2.set_ylabel("RMSE")
                ax2.set_title("RMSE Comparison")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # Table of results for the smallest and largest sample sizes
                st.subheader(f"Results for T = {min(sample_sizes)} and T = {max(sample_sizes)}")

                columns = ["Method", "Sample Size", "Mean Estimate", "Bias", "RMSE", "MAE"]
                table_data = []

                for method in correction_methods:
                    # Results for smallest sample size
                    small_result = results[method][0]
                    table_data.append([
                        method,
                        small_result["Sample Size"],
                        f"{small_result['Mean Estimate']:.4f}",
                        f"{small_result['Bias']:.4f}",
                        f"{small_result['RMSE']:.4f}",
                        f"{small_result['MAE']:.4f}"
                    ])

                    # Results for largest sample size
                    large_result = results[method][-1]
                    table_data.append([
                        method,
                        large_result["Sample Size"],
                        f"{large_result['Mean Estimate']:.4f}",
                        f"{large_result['Bias']:.4f}",
                        f"{large_result['RMSE']:.4f}",
                        f"{large_result['MAE']:.4f}"
                    ])

                st.table(pd.DataFrame(table_data, columns=columns))

                # Add explanations
                st.info("""
                **Explanation of Bias Correction Methods:**

                1. **Standard OLS:** The conventional estimator suffers from downward bias in small samples, with bias approximately -1/T.

                2. **Analytical correction:** Applies a formula-based correction that accounts for the small sample bias. Common methods 
                include Kendall's correction and median-unbiased estimation.

                3. **Bootstrap correction:** Uses bootstrap simulation to estimate the bias, then subtracts this estimated bias 
                from the original estimate.

                4. **Indirect inference:** Calibrates the model parameter so that simulated data from the model produces the same 
                biased estimate as observed in the actual data.
                """)

            elif model_type == "Binary choice (Logit/Probit)":
                # Binary choice model simulations
                for n in sample_sizes:
                    method_results = {method: [] for method in correction_methods}

                    for _ in range(n_simulations):
                        # Generate data
                        x = np.random.normal(0, 1, n)

                        if model_variant == "Logit":
                            p = 1 / (1 + np.exp(-true_beta_binary * x))
                            y = np.random.binomial(1, p, n)

                        else:  # Probit
                            p = stats.norm.cdf(true_beta_binary * x)
                            y = np.random.binomial(1, p, n)

                        # Estimation methods
                        X = sm.add_constant(x)

                        for method in correction_methods:
                            if method == "Standard ML":
                                try:
                                    if model_variant == "Logit":
                                        model = sm.Logit(y, X).fit(disp=0)
                                    else:
                                        model = sm.Probit(y, X).fit(disp=0)

                                    estimate = model.params[1]
                                except:
                                    # Skip if estimation fails
                                    continue

                            elif method == "Firth's penalized ML":
                                # Simplified version of Firth's method
                                try:
                                    if model_variant == "Logit":
                                        # Standard ML first
                                        model = sm.Logit(y, X).fit(disp=0)

                                        # Extract X and calculate hat matrix
                                        X_matrix = model.model.exog
                                        W = np.diag(model.predict() * (1 - model.predict()))
                                        H = X_matrix @ np.linalg.inv(X_matrix.T @ W @ X_matrix) @ X_matrix.T @ W

                                        # Add penalty term
                                        h = np.diag(H)
                                        y_mod = y + 0.5 * h * (1 - 2 * y)

                                        # Re-estimate with modified y
                                        model_firth = sm.Logit(y_mod, X).fit(disp=0)
                                        estimate = model_firth.params[1]
                                    else:
                                        # For Probit, use standard ML (simplification)
                                        model = sm.Probit(y, X).fit(disp=0)
                                        estimate = model.params[1]
                                except:
                                    # Skip if estimation fails
                                    continue

                            elif method == "Jackknife correction":
                                try:
                                    # Standard ML first
                                    if model_variant == "Logit":
                                        model = sm.Logit(y, X).fit(disp=0)
                                    else:
                                        model = sm.Probit(y, X).fit(disp=0)

                                    full_estimate = model.params[1]

                                    # Jackknife estimates
                                    jackknife_estimates = []

                                    # Limit number of leave-one-out samples for speed
                                    indices = np.random.choice(n, size=min(n, 20), replace=False)

                                    for i in indices:
                                        mask = np.ones(n, dtype=bool)
                                        mask[i] = False

                                        X_reduced = X[mask]
                                        y_reduced = y[mask]

                                        try:
                                            if model_variant == "Logit":
                                                model_jack = sm.Logit(y_reduced, X_reduced).fit(disp=0)
                                            else:
                                                model_jack = sm.Probit(y_reduced, X_reduced).fit(disp=0)

                                            jackknife_estimates.append(model_jack.params[1])
                                        except:
                                            # Skip if estimation fails
                                            continue

                                    # Calculate jackknife bias
                                    if jackknife_estimates:
                                        jack_bias = (len(jackknife_estimates) - 1) * (
                                                    np.mean(jackknife_estimates) - full_estimate)
                                        estimate = full_estimate - jack_bias
                                    else:
                                        estimate = full_estimate
                                except:
                                    # Skip if estimation fails
                                    continue

                            method_results[method].append(estimate)

                    # Calculate summary statistics
                    for method in correction_methods:
                        if method_results[method]:
                            estimates = method_results[method]
                            mean_est = np.mean(estimates)
                            bias = mean_est - true_beta_binary
                            rmse = np.sqrt(np.mean((np.array(estimates) - true_beta_binary) ** 2))

                            results[method].append({
                                "Sample Size": n,
                                "Mean Estimate": mean_est,
                                "Bias": bias,
                                "RMSE": rmse,
                                "MAE": np.mean(np.abs(np.array(estimates) - true_beta_binary))
                            })
                        else:
                            # If no successful estimations
                            results[method].append({
                                "Sample Size": n,
                                "Mean Estimate": np.nan,
                                "Bias": np.nan,
                                "RMSE": np.nan,
                                "MAE": np.nan
                            })

                # Create comparison plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Plot bias
                for method in correction_methods:
                    bias_values = [result["Bias"] for result in results[method]]
                    ax1.plot(sample_sizes, bias_values, 'o-', label=method)

                ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7)
                ax1.set_xlabel("Sample Size (n)")
                ax1.set_ylabel("Bias")
                ax1.set_title(f"Bias Comparison for {model_variant}, β = {true_beta_binary}")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot RMSE
                for method in correction_methods:
                    rmse_values = [result["RMSE"] for result in results[method]]
                    ax2.plot(sample_sizes, rmse_values, 'o-', label=method)

                ax2.set_xlabel("Sample Size (n)")
                ax2.set_ylabel("RMSE")
                ax2.set_title("RMSE Comparison")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # Table of results for the smallest and largest sample sizes
                st.subheader(f"Results for n = {min(sample_sizes)} and n = {max(sample_sizes)}")

                columns = ["Method", "Sample Size", "Mean Estimate", "Bias", "RMSE", "MAE"]
                table_data = []

                for method in correction_methods:
                    # Results for smallest sample size
                    small_result = results[method][0]
                    table_data.append([
                        method,
                        small_result["Sample Size"],
                        f"{small_result['Mean Estimate']:.4f}" if not np.isnan(
                            small_result['Mean Estimate']) else "N/A",
                        f"{small_result['Bias']:.4f}" if not np.isnan(small_result['Bias']) else "N/A",
                        f"{small_result['RMSE']:.4f}" if not np.isnan(small_result['RMSE']) else "N/A",
                        f"{small_result['MAE']:.4f}" if not np.isnan(small_result['MAE']) else "N/A"
                    ])

                    # Results for largest sample size
                    large_result = results[method][-1]
                    table_data.append([
                        method,
                        large_result["Sample Size"],
                        f"{large_result['Mean Estimate']:.4f}" if not np.isnan(
                            large_result['Mean Estimate']) else "N/A",
                        f"{large_result['Bias']:.4f}" if not np.isnan(large_result['Bias']) else "N/A",
                        f"{large_result['RMSE']:.4f}" if not np.isnan(large_result['RMSE']) else "N/A",
                        f"{large_result['MAE']:.4f}" if not np.isnan(large_result['MAE']) else "N/A"
                    ])

                st.table(pd.DataFrame(table_data, columns=columns))

                # Add explanations
                if model_variant == "Logit":
                    st.info("""
                    **Explanation of Bias Correction Methods for Logit:**

                    1. **Standard ML:** Maximum Likelihood estimator has O(1/n) bias in small samples and can suffer from 
                    separation problems where certain combinations of predictors perfectly predict the outcome.

                    2. **Firth's penalized ML:** Adds a penalty term to the likelihood function that reduces small-sample bias. 
                    This method is particularly useful for dealing with separation issues in small samples.

                    3. **Jackknife correction:** Uses the leave-one-out method to estimate and correct for bias, reducing 
                    the impact of influential observations.
                    """)
                else:
                    st.info("""
                    **Explanation of Bias Correction Methods for Probit:**

                    1. **Standard ML:** The Probit ML estimator has bias of order O(1/n) in small samples.

                    2. **Firth's penalized ML:** While originally developed for logistic regression, similar penalization 
                    approaches can be applied to Probit models to reduce small-sample bias.

                    3. **Jackknife correction:** Uses leave-one-out resampling to estimate and correct for bias in the 
                    Probit coefficient estimates.
                    """)

            else:  # Panel data simulations
                # Limited implementation for panel data due to complexity
                st.markdown("### Panel Data Simulation Results")

                if panel_type == "Static panel":
                    st.info("""
                    **Static Panel Data Models and Small Sample Issues:**

                    1. **Pooled OLS:** Ignores panel structure but provides consistent estimates if individual effects are uncorrelated with regressors

                    2. **Fixed Effects:** Controls for time-invariant heterogeneity but consumes degrees of freedom in small T samples

                    Simulation Results:
                    - With small N and small T, fixed effects models suffer from incidental parameters problem
                    - As T increases, the bias in fixed effects diminishes at rate O(1/T)
                    - With very small T (e.g., T=2 or T=3), random effects may be preferable despite potential specification issues
                    """)

                    # Static panel data visualization
                    fig, ax = plt.subplots(figsize=(10, 6))

                    T_values = [pair[1] for pair in sample_sizes]

                    # Theoretical bias for pooled OLS (assuming correlation between xi and regressors)
                    bias_pooled = [0.3] * len(T_values)  # Fixed bias regardless of T

                    # Theoretical bias for fixed effects (approximately 1/T bias)
                    bias_fe = [0.6 / t for t in T_values]

                    ax.plot(T_values, bias_pooled, 'ro-', label="Pooled OLS Bias")
                    ax.plot(T_values, bias_fe, 'bs-', label="Fixed Effects Bias")
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)

                    ax.set_xlabel("Time Periods (T)")
                    ax.set_ylabel("Approximate Bias")
                    ax.set_title("Theoretical Bias in Static Panel Models")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)

                else:  # Dynamic panel
                    st.warning("""
                    **Dynamic Panel Data Models and Small Sample Issues:**

                    1. **Nickell Bias:** Fixed effects estimator in dynamic panels has bias of order O(1/T)

                    2. **First-Difference & Arellano-Bond GMM:** Eliminates fixed effects through first-differencing, 
                    then uses lagged levels as instruments. Performs poorly in small T with persistent series.

                    3. **System GMM (Blundell-Bond):** Adds additional moment conditions using lagged differences as 
                    instruments for levels. More efficient with persistent series but requires additional assumptions.

                    Simulation Results:
                    - For T < 10, bias in standard fixed effects can be substantial
                    - Arellano-Bond estimator can have high variance in small samples
                    - System GMM generally performs better with persistent series but requires more instruments
                    - As N increases, all estimators improve, but bias remains in fixed effects regardless of N when T is small
                    """)

                    # Dynamic panel data visualization
                    fig, ax = plt.subplots(figsize=(10, 6))

                    T_values = [pair[1] for pair in sample_sizes]

                    # Theoretical bias for different estimators in dynamic panel
                    true_lambda = 0.7  # Assumed value

                    # Nickell bias formula: approximately -(1+lambda)/(T-1)
                    bias_fe = [-(1 + true_lambda) / (t - 1) for t in T_values]

                    # Simplified bias for other methods (illustrative)
                    bias_abond = [0.1 / t for t in T_values]  # Decreasing with T
                    bias_bbond = [0.05 / t for t in T_values]  # Smaller bias

                    ax.plot(T_values, bias_fe, 'ro-', label="Fixed Effects Bias")
                    ax.plot(T_values, bias_abond, 'bs-', label="Arellano-Bond Bias")
                    ax.plot(T_values, bias_bbond, 'gd-', label="Blundell-Bond Bias")
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)

                    ax.set_xlabel("Time Periods (T)")
                    ax.set_ylabel("Approximate Bias")
                    ax.set_title("Theoretical Bias in Dynamic Panel Models (λ = 0.7)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    st.pyplot(fig)

                st.info("""
                **Resources for Panel Data Small Sample Issues:**

                1. **Analytical Bias Corrections:**
                   - Half-panel jackknife for fixed effects models
                   - Split-panel jackknife for dynamic models
                   - Recursive mean adjustment methods

                2. **Alternative Estimators:**
                   - Kiviet bias-corrected LSDV for dynamic panels
                   - Indirect inference for dynamic models
                   - Penalized maximum likelihood methods

                3. **Inference Approaches:**
                   - Cluster-robust standard errors (unreliable with few clusters)
                   - Bootstrap methods appropriate for panel structure
                   - Randomization/permutation tests
                """)

###########################################
# PAGE: Time Series Challenges
###########################################
elif selected_page == "Time Series Challenges":
    st.title("Time Series Challenges in Small Samples")

    st.markdown("""
    Time series analysis is particularly affected by small sample issues due to the sequential nature
    of the data. This section explores specific challenges in time series econometrics with limited observations.
    """)

    tab1, tab2, tab3 = st.tabs(["Autoregressive Models", "Unit Root Testing", "Forecasting"])

    with tab1:
        st.subheader("Autoregressive Models in Small Samples")

        st.markdown("""
        Autoregressive (AR) models are fundamental in time series econometrics, but they face several
        challenges when estimated with limited time periods. This tool explores these issues.
        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            ar_order = st.radio(
                "AR model order:",
                ["AR(1)", "AR(2)"]
            )

            if ar_order == "AR(1)":
                true_ar1 = st.slider(
                    "True AR(1) coefficient (ρ)",
                    min_value=-0.9,
                    max_value=0.9,
                    value=0.7,
                    step=0.1,
                    key="ar1_coef"
                )

                model_params = {"ar1": true_ar1}

            else:  # AR(2)
                true_ar1 = st.slider(
                    "True AR(1) coefficient (ρ₁)",
                    min_value=-0.9,
                    max_value=0.9,
                    value=0.5,
                    step=0.1,
                    key="ar2_coef1"
                )

                true_ar2 = st.slider(
                    "True AR(2) coefficient (ρ₂)",
                    min_value=-0.9,
                    max_value=0.9,
                    value=0.3,
                    step=0.1,
                    key="ar2_coef2"
                )

                # Check stationarity condition
                if abs(true_ar1 + true_ar2) >= 1 or abs(true_ar2) >= 1 or abs(true_ar1 - true_ar2) >= 1:
                    st.warning("Warning: Selected coefficients may not satisfy stationarity conditions.")

                model_params = {"ar1": true_ar1, "ar2": true_ar2}

            estimation_method = st.radio(
                "Estimation method:",
                ["OLS", "Yule-Walker", "Maximum Likelihood", "Bayesian (with prior)"]
            )

            if estimation_method == "Bayesian (with prior)":
                prior_strength = st.slider(
                    "Prior strength (0 = uninformative, 1 = strong)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1
                )

                prior_mean = st.slider(
                    "Prior mean for AR coefficient",
                    min_value=-0.5,
                    max_value=0.9,
                    value=0.0,
                    step=0.1
                )

            n_simulations = st.slider(
                "Number of simulations",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="ar_simulations"
            )

            # Show a series visualization
            st.subheader("Sample AR Process")

            # Generate a sample AR process with specified parameters
            T_sample = 50
            np.random.seed(42)

            sample_series = np.zeros(T_sample)
            e = np.random.normal(0, 1, T_sample)

            if ar_order == "AR(1)":
                for t in range(1, T_sample):
                    sample_series[t] = true_ar1 * sample_series[t - 1] + e[t]
            else:  # AR(2)
                for t in range(2, T_sample):
                    sample_series[t] = true_ar1 * sample_series[t - 1] + true_ar2 * sample_series[t - 2] + e[t]

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(sample_series)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title(f"Sample {ar_order} Process")
            st.pyplot(fig)

        with col2:
            # Range of sample sizes to analyze
            sample_sizes = [10, 20, 30, 50, 100, 200]

            # Store results for each sample size
            ar_results = []

            # Run simulations for each sample size
            for T in sample_sizes:
                ar1_estimates = []
                ar2_estimates = [] if ar_order == "AR(2)" else None

                for _ in range(n_simulations):
                    # Generate AR process
                    y = np.zeros(T)
                    e = np.random.normal(0, 1, T)

                    if ar_order == "AR(1)":
                        for t in range(1, T):
                            y[t] = true_ar1 * y[t - 1] + e[t]

                        # Skip initial observations for burn-in
                        burn_in = min(10, T // 4)
                        y = y[burn_in:]
                        T_effective = len(y)

                        # Prepare data for estimation
                        y_lag = y[:-1]
                        y_current = y[1:]

                        if estimation_method == "OLS":
                            # Use OLS regression
                            model = sm.OLS(y_current, y_lag).fit()
                            ar1_est = model.params[0]

                        elif estimation_method == "Yule-Walker":
                            # Use Yule-Walker equations (via statsmodels)
                            model = sm.tsa.ar_model.AutoReg(y, lags=1).fit(method='yule_walker')
                            ar1_est = model.params[1]  # First coefficient

                        elif estimation_method == "Maximum Likelihood":
                            # Use MLE estimation
                            model = sm.tsa.ar_model.AutoReg(y, lags=1).fit(method='mle')
                            ar1_est = model.params[1]  # First coefficient

                        elif estimation_method == "Bayesian (with prior)":
                            # Simplified Bayesian approach with conjugate prior
                            # Prior: N(prior_mean, (1/prior_strength) * σ²/SSx)
                            var_y = np.var(y_current)
                            SSx = np.sum((y_lag - np.mean(y_lag)) ** 2)

                            # OLS estimate
                            beta_ols = np.sum(y_lag * y_current) / SSx

                            # Posterior mean (simplified)
                            precision_prior = prior_strength * SSx / var_y
                            precision_ols = SSx / var_y

                            ar1_est = (precision_prior * prior_mean + precision_ols * beta_ols) / (
                                        precision_prior + precision_ols)

                        ar1_estimates.append(ar1_est)

                    else:  # AR(2)
                        for t in range(2, T):
                            y[t] = true_ar1 * y[t - 1] + true_ar2 * y[t - 2] + e[t]

                        # Skip initial observations for burn-in
                        burn_in = min(10, T // 4)
                        y = y[burn_in:]
                        T_effective = len(y)

                        if T_effective <= 3:
                            # Not enough data for AR(2) estimation after burn-in
                            continue

                        if estimation_method == "OLS":
                            # Prepare data for AR(2) estimation
                            y_current = y[2:]
                            y_lag1 = y[1:-1]
                            y_lag2 = y[:-2]

                            X = np.column_stack((y_lag1, y_lag2))

                            # Use OLS regression
                            model = sm.OLS(y_current, X).fit()
                            ar1_est, ar2_est = model.params

                        elif estimation_method in ["Yule-Walker", "Maximum Likelihood"]:
                            # Use statsmodels
                            method = 'yule_walker' if estimation_method == "Yule-Walker" else 'mle'
                            try:
                                model = sm.tsa.ar_model.AutoReg(y, lags=2).fit(method=method)
                                ar1_est = model.params[1]  # First coefficient
                                ar2_est = model.params[2]  # Second coefficient
                            except:
                                # Skip if estimation fails
                                continue

                        elif estimation_method == "Bayesian (with prior)":
                            # Simplified approach - apply prior to both coefficients equally
                            # Real Bayesian analysis would use proper multivariate priors

                            # Prepare data
                            y_current = y[2:]
                            y_lag1 = y[1:-1]
                            y_lag2 = y[:-2]

                            X = np.column_stack((y_lag1, y_lag2))

                            # OLS estimates
                            model = sm.OLS(y_current, X).fit()
                            beta_ols = model.params

                            # Apply shrinkage toward prior mean
                            ar1_est = beta_ols[0] * (1 - prior_strength) + prior_mean * prior_strength
                            ar2_est = beta_ols[1] * (1 - prior_strength) + prior_mean * prior_strength

                        ar1_estimates.append(ar1_est)
                        ar2_estimates.append(ar2_est)

                # Calculate summary statistics
                if ar_order == "AR(1)":
                    mean_ar1 = np.mean(ar1_estimates)
                    bias_ar1 = mean_ar1 - true_ar1
                    rmse_ar1 = np.sqrt(np.mean((np.array(ar1_estimates) - true_ar1) ** 2))

                    ar_results.append({
                        "Sample Size": T,
                        "Mean AR(1)": mean_ar1,
                        "Bias AR(1)": bias_ar1,
                        "RMSE AR(1)": rmse_ar1,
                        "Std Dev AR(1)": np.std(ar1_estimates)
                    })

                else:  # AR(2)
                    if ar1_estimates and ar2_estimates:  # Check if we have estimates
                        mean_ar1 = np.mean(ar1_estimates)
                        mean_ar2 = np.mean(ar2_estimates)
                        bias_ar1 = mean_ar1 - true_ar1
                        bias_ar2 = mean_ar2 - true_ar2
                        rmse_ar1 = np.sqrt(np.mean((np.array(ar1_estimates) - true_ar1) ** 2))
                        rmse_ar2 = np.sqrt(np.mean((np.array(ar2_estimates) - true_ar2) ** 2))

                        ar_results.append({
                            "Sample Size": T,
                            "Mean AR(1)": mean_ar1,
                            "Mean AR(2)": mean_ar2,
                            "Bias AR(1)": bias_ar1,
                            "Bias AR(2)": bias_ar2,
                            "RMSE AR(1)": rmse_ar1,
                            "RMSE AR(2)": rmse_ar2
                        })

            # Create bias plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract data for plotting
            T_values = [result["Sample Size"] for result in ar_results]

            if ar_order == "AR(1)":
                bias_values = [result["Bias AR(1)"] for result in ar_results]
                rmse_values = [result["RMSE AR(1)"] for result in ar_results]

                # Plot bias
                ax.plot(T_values, bias_values, 'b-o', linewidth=2, label="Bias")
                ax.plot(T_values, rmse_values, 'r-^', linewidth=2, label="RMSE")

                # Theoretical Hurwicz bias = -(1+3ρ)/T for AR(1)
                theoretical_bias = [-(1 + 3 * true_ar1) / T for T in T_values]
                ax.plot(T_values, theoretical_bias, 'g--', linewidth=2, label="Theoretical Bias")

                ax.set_title(f"Bias and RMSE for AR(1) Coefficient (ρ = {true_ar1}), {estimation_method} Estimation")

            else:  # AR(2)
                bias_ar1_values = [result["Bias AR(1)"] for result in ar_results]
                bias_ar2_values = [result["Bias AR(2)"] for result in ar_results]
                rmse_ar1_values = [result["RMSE AR(1)"] for result in ar_results]
                rmse_ar2_values = [result["RMSE AR(2)"] for result in ar_results]

                # Plot bias for AR(1) coefficient
                ax.plot(T_values, bias_ar1_values, 'b-o', linewidth=2, label="Bias AR(1)")
                ax.plot(T_values, rmse_ar1_values, 'r-^', linewidth=2, label="RMSE AR(1)")

                # Plot bias for AR(2) coefficient (dashed lines)
                ax.plot(T_values, bias_ar2_values, 'b--o', linewidth=2, label="Bias AR(2)")
                ax.plot(T_values, rmse_ar2_values, 'r--^', linewidth=2, label="RMSE AR(2)")

                ax.set_title(
                    f"Bias and RMSE for AR(2) Coefficients (ρ₁ = {true_ar1}, ρ₂ = {true_ar2}), {estimation_method} Estimation")

            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel("Sample Size (T)")
            ax.set_ylabel("Bias / RMSE")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Create distribution plot for different sample sizes
            st.subheader("Sampling Distribution of Coefficients")

            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # Choose a subset of sample sizes to display
            display_sizes = [min(sample_sizes), sample_sizes[len(sample_sizes) // 2], max(sample_sizes)]

            # For each chosen sample size, run a new simulation to get estimates
            for T in display_sizes:
                ar1_estimates = []

                for _ in range(n_simulations):
                    # Generate AR process
                    y = np.zeros(T)
                    e = np.random.normal(0, 1, T)

                    if ar_order == "AR(1)":
                        for t in range(1, T):
                            y[t] = true_ar1 * y[t - 1] + e[t]

                        # Skip initial observations for burn-in
                        burn_in = min(10, T // 4)
                        y = y[burn_in:]

                        # Prepare data for estimation
                        y_lag = y[:-1]
                        y_current = y[1:]

                        if estimation_method == "OLS":
                            model = sm.OLS(y_current, y_lag).fit()
                            ar1_est = model.params[0]
                        elif estimation_method == "Yule-Walker":
                            model = sm.tsa.ar_model.AutoReg(y, lags=1).fit(method='yule_walker')
                            ar1_est = model.params[1]
                        elif estimation_method == "Maximum Likelihood":
                            model = sm.tsa.ar_model.AutoReg(y, lags=1).fit(method='mle')
                            ar1_est = model.params[1]
                        elif estimation_method == "Bayesian (with prior)":
                            var_y = np.var(y_current)
                            SSx = np.sum((y_lag - np.mean(y_lag)) ** 2)
                            beta_ols = np.sum(y_lag * y_current) / SSx
                            precision_prior = prior_strength * SSx / var_y
                            precision_ols = SSx / var_y
                            ar1_est = (precision_prior * prior_mean + precision_ols * beta_ols) / (
                                        precision_prior + precision_ols)

                        ar1_estimates.append(ar1_est)

                # Plot density for this sample size
                sns.kdeplot(ar1_estimates, label=f"T = {T}", ax=ax2)

            # Add vertical line for true parameter
            if ar_order == "AR(1)":
                ax2.axvline(x=true_ar1, color='k', linestyle='--', label=f"True ρ = {true_ar1}")
                ax2.set_xlabel("Estimated AR(1) Coefficient")
            else:  # For AR(2), we just show AR(1) coefficient for simplicity
                ax2.axvline(x=true_ar1, color='k', linestyle='--', label=f"True ρ₁ = {true_ar1}")
                ax2.set_xlabel("Estimated AR(1) Coefficient (ρ₁)")

            ax2.set_ylabel("Density")
            ax2.set_title(f"Sampling Distribution for Different Sample Sizes ({estimation_method} Estimation)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig2)

            # Create a table of results
            st.subheader("Numerical Results")
            df_results = pd.DataFrame(ar_results)
            st.dataframe(df_results)

            st.markdown("""
                            ### Key Observations:

                            1. **Downward Bias**: Small samples typically exhibit downward bias (Hurwicz bias) in AR estimates.
                            2. **Increased Variance**: Sampling variability is much higher in small samples.
                            3. **Estimation Method Matters**: Different estimation approaches can mitigate bias differently.
                            4. **Persistence Amplifies Bias**: Higher true AR coefficients lead to more severe small-sample bias.
                            5. **Rapid Improvement**: Bias and RMSE typically improve rapidly as sample size increases from very small (T<30) to moderate.
                            """)

        with tab2:
            st.subheader("Unit Root Testing in Small Samples")

            st.markdown("""
                        Unit root tests are crucial for determining stationarity, but they often have low power in small samples.
                        This tool demonstrates the challenges with the Augmented Dickey-Fuller (ADF) test.
                        """)

            col1, col2 = st.columns([1, 2])

            with col1:
                dgt_process = st.radio(
                    "Data generating process:",
                    ["Pure random walk (Unit root)", "Near unit root (ρ = 0.95)", "Stationary AR(1)"]
                )

                if dgt_process == "Pure random walk (Unit root)":
                    true_ar_coef = 1.0
                elif dgt_process == "Near unit root (ρ = 0.95)":
                    true_ar_coef = 0.95
                else:  # Stationary AR(1)
                    true_ar_coef = st.slider(
                        "AR coefficient (ρ)",
                        min_value=0.0,
                        max_value=0.9,
                        value=0.7,
                        step=0.1
                    )

                include_trend = st.checkbox("Include trend in ADF test", value=False)

                n_adf_simulations = st.slider(
                    "Number of simulations",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=100
                )

                # Generate a sample path for illustration
                st.subheader("Sample Path")

                T_sample = 100
                np.random.seed(42)

                e = np.random.normal(0, 1, T_sample)
                sample_path = np.zeros(T_sample)

                for t in range(1, T_sample):
                    sample_path[t] = true_ar_coef * sample_path[t - 1] + e[t]

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(sample_path)
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.set_title(f"Sample Path (ρ = {true_ar_coef})")
                st.pyplot(fig)

            with col2:
                # Sample sizes to analyze
                sample_sizes = [20, 30, 50, 100, 200, 500]

                # Store rejection rates
                rejection_rates = []

                # Set significance level
                alpha = 0.05

                # Run simulations
                for T in sample_sizes:
                    rejections = 0

                    for _ in range(n_adf_simulations):
                        # Generate process
                        y = np.zeros(T)
                        e = np.random.normal(0, 1, T)

                        for t in range(1, T):
                            y[t] = true_ar_coef * y[t - 1] + e[t]

                        # Perform ADF test
                        trend = 'ct' if include_trend else 'c'
                        result = sm.tsa.stattools.adfuller(y, regression=trend)

                        # Check if null hypothesis (unit root exists) is rejected
                        p_value = result[1]
                        if p_value < alpha:
                            rejections += 1

                    # Calculate rejection rate
                    rejection_rate = rejections / n_adf_simulations
                    rejection_rates.append(rejection_rate)

                # Create plot for rejection rates
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot rejection rates
                ax.plot(sample_sizes, rejection_rates, 'b-o', linewidth=2)

                # Add horizontal line for significance level
                ax.axhline(y=alpha, color='r', linestyle='--', label=f"Nominal size (α = {alpha})")

                # Interpretation depends on true DGP
                if dgt_process == "Pure random walk (Unit root)":
                    ax.set_title("Size of ADF Test (Rejection Rate Under True Unit Root)")
                    st.markdown("""
                                **Interpretation**: When the null hypothesis is true (there is a unit root), 
                                the rejection rate should equal the significance level (α). Higher values indicate 
                                size distortion (over-rejection).
                                """)
                else:
                    ax.set_title("Power of ADF Test (Rejection Rate Under Stationarity)")
                    st.markdown("""
                                **Interpretation**: When the null hypothesis is false (no unit root), 
                                the rejection rate represents the power of the test. Higher values indicate 
                                better ability to correctly reject the false null hypothesis.
                                """)

                ax.set_xlabel("Sample Size (T)")
                ax.set_ylabel("Rejection Rate")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Create table of results
                st.subheader("Rejection Rates by Sample Size")
                results_df = pd.DataFrame({
                    "Sample Size": sample_sizes,
                    "Rejection Rate": rejection_rates
                })
                st.table(results_df)

                # Add interpretations
                if true_ar_coef == 1.0:
                    st.markdown("""
                                ### Key Observations for Unit Root Process:

                                1. **Size Distortion**: In small samples, the ADF test often rejects the null hypothesis more frequently 
                                   than the nominal significance level would suggest (size distortion).
                                2. **Trend Inclusion**: Including a trend in the test equation when none exists in the DGP 
                                   can exacerbate size distortions.
                                3. **Approach to Nominal Size**: As sample size increases, the actual size approaches the nominal level.
                                """)
                elif true_ar_coef >= 0.9:
                    st.markdown("""
                                ### Key Observations for Near Unit Root Process:

                                1. **Low Power**: The ADF test has very low power against near-unit-root alternatives in small samples.
                                2. **Slow Improvement**: Power increases with sample size, but very slowly for processes close to unit root.
                                3. **Practical Challenge**: For typical macroeconomic time series (often 50-100 observations), 
                                   distinguishing between unit root and near-unit-root processes is extremely difficult.
                                """)
                else:
                    st.markdown("""
                                ### Key Observations for Stationary Process:

                                1. **Sample Size Dependency**: Power increases with sample size - larger samples make it easier 
                                   to correctly reject the false null hypothesis.
                                2. **Persistence Effects**: More persistent processes (higher AR coefficients) require larger 
                                   samples for reliable inference.
                                3. **Deterministic Components**: Test specification (inclusion of constant, trend) affects power.
                                """)

        with tab3:
            st.subheader("Forecasting Accuracy in Small Samples")

            st.markdown("""
                        Forecasting with time series models estimated on small samples introduces additional uncertainty.
                        This tool explores how forecast accuracy and uncertainty depend on sample size.
                        """)

            col1, col2 = st.columns([1, 2])

            with col1:
                forecast_model = st.radio(
                    "Forecasting model:",
                    ["AR(1)", "AR(2)", "ARIMA(1,1,0)"]
                )

                if forecast_model == "AR(1)":
                    true_ar_param = st.slider(
                        "True AR(1) coefficient",
                        min_value=0.0,
                        max_value=0.9,
                        value=0.7,
                        step=0.1,
                        key="forecast_ar1"
                    )

                    true_params = {"ar1": true_ar_param}

                elif forecast_model == "AR(2)":
                    true_ar1 = st.slider(
                        "True AR(1) coefficient",
                        min_value=0.0,
                        max_value=0.8,
                        value=0.5,
                        step=0.1,
                        key="forecast_ar21"
                    )

                    true_ar2 = st.slider(
                        "True AR(2) coefficient",
                        min_value=0.0,
                        max_value=0.4,
                        value=0.3,
                        step=0.1,
                        key="forecast_ar22"
                    )

                    true_params = {"ar1": true_ar1, "ar2": true_ar2}

                else:  # ARIMA(1,1,0)
                    true_ar_param = st.slider(
                        "True AR coefficient (after differencing)",
                        min_value=0.0,
                        max_value=0.7,
                        value=0.4,
                        step=0.1,
                        key="forecast_arima"
                    )

                    true_params = {"ar1": true_ar_param}

                forecast_horizon = st.slider(
                    "Forecast horizon",
                    min_value=1,
                    max_value=12,
                    value=6,
                    step=1
                )

                n_forecast_simulations = st.slider(
                    "Number of simulations",
                    min_value=100,
                    max_value=1000,
                    value=500,
                    step=100,
                    key="forecast_sims"
                )

                # Visualize a sample forecast
                st.subheader("Sample Forecast")

                # Generate a sample time series
                T_sample = 50
                np.random.seed(42)

                sample_series = np.zeros(T_sample + forecast_horizon)
                e = np.random.normal(0, 1, T_sample + forecast_horizon)

                if forecast_model == "AR(1)":
                    for t in range(1, T_sample + forecast_horizon):
                        sample_series[t] = true_params["ar1"] * sample_series[t - 1] + e[t]

                    # Fit AR(1) model to in-sample data
                    model = sm.tsa.ar_model.AutoReg(sample_series[:T_sample], lags=1).fit()

                    # Generate forecasts
                    forecasts = model.predict(start=T_sample, end=T_sample + forecast_horizon - 1)

                elif forecast_model == "AR(2)":
                    for t in range(2, T_sample + forecast_horizon):
                        sample_series[t] = true_params["ar1"] * sample_series[t - 1] + true_params["ar2"] * \
                                           sample_series[t - 2] + e[t]

                    # Fit AR(2) model to in-sample data
                    model = sm.tsa.ar_model.AutoReg(sample_series[:T_sample], lags=2).fit()

                    # Generate forecasts
                    forecasts = model.predict(start=T_sample, end=T_sample + forecast_horizon - 1)

                else:  # ARIMA(1,1,0)
                    # Generate integrated series
                    for t in range(1, T_sample + forecast_horizon):
                        sample_series[t] = sample_series[t - 1] + true_params["ar1"] * (
                                    sample_series[t - 1] - (0 if t - 2 < 0 else sample_series[t - 2])) + e[t]

                    # Fit ARIMA model to in-sample data
                    model = ARIMA(sample_series[:T_sample], order=(1, 1, 0)).fit()

                    # Generate forecasts
                    forecasts = model.forecast(steps=forecast_horizon)

                # Plot the sample and forecasts
                fig, ax = plt.subplots(figsize=(8, 4))

                # Plot historical data
                ax.plot(np.arange(T_sample), sample_series[:T_sample], 'b-', label='Historical Data')

                # Plot true future values
                ax.plot(np.arange(T_sample, T_sample + forecast_horizon), sample_series[T_sample:], 'g--',
                        label='True Future Values')

                # Plot forecasts
                ax.plot(np.arange(T_sample, T_sample + forecast_horizon), forecasts, 'r-', label='Forecasts')

                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.set_title(f"Sample Forecast with {forecast_model}")
                ax.axvline(x=T_sample - 1, color='k', linestyle='--', alpha=0.3)
                ax.legend()
                st.pyplot(fig)

            with col2:
                # Sample sizes to analyze
                sample_sizes = [20, 30, 50, 100, 200]

                # Store forecast errors
                forecast_results = []

                # Run simulations
                for T in sample_sizes:
                    mse_values = np.zeros(forecast_horizon)
                    mae_values = np.zeros(forecast_horizon)
                    ci_width_values = np.zeros(forecast_horizon)

                    for _ in range(n_forecast_simulations):
                        # Generate time series with T+forecast_horizon observations
                        y = np.zeros(T + forecast_horizon)
                        e = np.random.normal(0, 1, T + forecast_horizon)

                        if forecast_model == "AR(1)":
                            for t in range(1, T + forecast_horizon):
                                y[t] = true_params["ar1"] * y[t - 1] + e[t]

                            # Fit AR(1) model to in-sample data
                            try:
                                model = sm.tsa.ar_model.AutoReg(y[:T], lags=1).fit()

                                # Generate forecasts and prediction intervals
                                forecasts = model.predict(start=T, end=T + forecast_horizon - 1)
                                pred_intervals = model.get_prediction(start=T, end=T + forecast_horizon - 1).conf_int(
                                    alpha=0.05)

                                # Calculate CI width for each horizon
                                for h in range(forecast_horizon):
                                    if h < len(forecasts):
                                        ci_width_values[h] += pred_intervals[h, 1] - pred_intervals[h, 0]

                            except:
                                # Skip if model fitting fails
                                continue

                        elif forecast_model == "AR(2)":
                            for t in range(2, T + forecast_horizon):
                                y[t] = true_params["ar1"] * y[t - 1] + true_params["ar2"] * y[t - 2] + e[t]

                            # Fit AR(2) model to in-sample data
                            try:
                                model = sm.tsa.ar_model.AutoReg(y[:T], lags=2).fit()

                                # Generate forecasts and prediction intervals
                                forecasts = model.predict(start=T, end=T + forecast_horizon - 1)
                                pred_intervals = model.get_prediction(start=T, end=T + forecast_horizon - 1).conf_int(
                                    alpha=0.05)

                                # Calculate CI width for each horizon
                                for h in range(forecast_horizon):
                                    if h < len(forecasts):
                                        ci_width_values[h] += pred_intervals[h, 1] - pred_intervals[h, 0]

                            except:
                                # Skip if model fitting fails
                                continue

                        else:  # ARIMA(1,1,0)
                            # Generate integrated series
                            for t in range(1, T + forecast_horizon):
                                y[t] = y[t - 1] + true_params["ar1"] * (y[t - 1] - (0 if t - 2 < 0 else y[t - 2])) + e[
                                    t]

                            # Fit ARIMA model to in-sample data
                            try:
                                model = ARIMA(y[:T], order=(1, 1, 0)).fit()

                                # Generate forecasts
                                forecasts = model.forecast(steps=forecast_horizon)
                                pred_intervals = model.get_forecast(steps=forecast_horizon).conf_int(alpha=0.05)

                                # Calculate CI width for each horizon
                                for h in range(forecast_horizon):
                                    if h < len(forecasts):
                                        ci_width_values[h] += pred_intervals.iloc[h, 1] - pred_intervals.iloc[h, 0]

                            except:
                                # Skip if model fitting fails
                                continue

                        # Calculate forecast errors for each horizon
                        for h in range(forecast_horizon):
                            if h < len(forecasts):
                                true_value = y[T + h]
                                forecast = forecasts[h]

                                # Calculate squared error and absolute error
                                squared_error = (forecast - true_value) ** 2
                                absolute_error = abs(forecast - true_value)

                                mse_values[h] += squared_error
                                mae_values[h] += absolute_error

                    # Calculate average MSE, MAE, and CI width for each horizon
                    mse_values /= n_forecast_simulations
                    mae_values /= n_forecast_simulations
                    ci_width_values /= n_forecast_simulations

                    # Store results
                    for h in range(forecast_horizon):
                        forecast_results.append({
                            "Sample Size": T,
                            "Horizon": h + 1,
                            "MSE": mse_values[h],
                            "MAE": mae_values[h],
                            "CI Width": ci_width_values[h]
                        })

                # Convert to DataFrame
                forecast_df = pd.DataFrame(forecast_results)

                # Create plots for forecast accuracy
                st.subheader("Forecast Accuracy by Sample Size and Horizon")

                # Plot RMSE by horizon for different sample sizes
                fig, ax = plt.subplots(figsize=(10, 6))

                for T in sample_sizes:
                    subset = forecast_df[forecast_df["Sample Size"] == T]
                    ax.plot(subset["Horizon"], np.sqrt(subset["MSE"]), 'o-', linewidth=2, label=f"T = {T}")

                ax.set_xlabel("Forecast Horizon")
                ax.set_ylabel("Root Mean Squared Error (RMSE)")
                ax.set_title(f"Forecast Accuracy for {forecast_model} Model")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Plot Confidence Interval Width by horizon for different sample sizes
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                for T in sample_sizes:
                    subset = forecast_df[forecast_df["Sample Size"] == T]
                    ax2.plot(subset["Horizon"], subset["CI Width"], 'o-', linewidth=2, label=f"T = {T}")

                ax2.set_xlabel("Forecast Horizon")
                ax2.set_ylabel("Width of 95% Prediction Interval")
                ax2.set_title(f"Forecast Uncertainty for {forecast_model} Model")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig2)

                # Table of results for h=1 forecasts across different sample sizes
                st.subheader("One-Step Ahead Forecast Performance")
                h1_results = forecast_df[forecast_df["Horizon"] == 1][["Sample Size", "RMSE", "MAE", "CI Width"]]
                h1_results["RMSE"] = np.sqrt(h1_results["MSE"])
                h1_results = h1_results[["Sample Size", "RMSE", "MAE", "CI Width"]]
                st.table(h1_results)

                st.markdown("""
                            ### Key Observations:

                            1. **Error Accumulation**: Forecast errors grow with horizon, but at different rates depending on sample size.
                            2. **Uncertainty Underestimation**: Small samples often underestimate forecast uncertainty.
                            3. **Rapid Improvement**: Forecast accuracy improves substantially with initial increases in sample size,
                               but with diminishing returns beyond a certain point.
                            4. **Horizon Impact**: The benefits of larger samples are particularly pronounced for longer forecast horizons.
                            5. **Model Complexity**: More complex models (higher-order AR, ARIMA) require larger samples for reliable forecasting.
                            """)

        ###########################################
        # PAGE: Model Selection Issues
        ###########################################
elif selected_page =="Model Selection Issues":
     st.title("Model Selection Issues in Small Samples")


     st.markdown("""
                    Model selection criteria like AIC, BIC, and cross-validation can behave differently with limited observations.
                    This section explores how sample size affects model selection decisions and the risks of overfitting.
                    """)

     tab1, tab2 = st.tabs(["Information Criteria", "Cross-Validation"])

     with tab1:
            st.subheader("Information Criteria Performance")

            st.markdown("""
                        This tool investigates how information criteria like AIC and BIC perform in model selection
                        with varying sample sizes. It compares their ability to identify the true model specification.
                        """)

            col1, col2 = st.columns([1, 2])

            with col1:
                true_model = st.radio(
                    "True data generating process:",
                    ["Linear (y = β₀ + β₁x₁ + ε)",
                     "Quadratic (y = β₀ + β₁x₁ + β₂x₁² + ε)",
                     "Multiple regression (y = β₀ + β₁x₁ + β₂x₂ + ε)"]
                )

                if true_model == "Linear (y = β₀ + β₁x₁ + ε)":
                    true_beta0 = st.slider("Intercept (β₀)", -5.0, 5.0, 1.0, 0.5)
                    true_beta1 = st.slider("Slope (β₁)", -5.0, 5.0, 2.0, 0.5)
                    true_params = [true_beta0, true_beta1]
                    true_order = 1

                elif true_model == "Quadratic (y = β₀ + β₁x₁ + β₂x₁² + ε)":
                    true_beta0 = st.slider("Intercept (β₀)", -5.0, 5.0, 1.0, 0.5)
                    true_beta1 = st.slider("Linear term (β₁)", -5.0, 5.0, 0.5, 0.5)
                    true_beta2 = st.slider("Quadratic term (β₂)", -2.0, 2.0, 1.0, 0.1)
                    true_params = [true_beta0, true_beta1, true_beta2]
                    true_order = 2

                else:  # Multiple regression
                    true_beta0 = st.slider("Intercept (β₀)", -5.0, 5.0, 0.5, 0.5)
                    true_beta1 = st.slider("Coefficient for x₁ (β₁)", -5.0, 5.0, 2.0, 0.5)
                    true_beta2 = st.slider("Coefficient for x₂ (β₂)", -5.0, 5.0, 1.5, 0.5)
                    true_params = [true_beta0, true_beta1, true_beta2]
                    true_order = "multi"

                noise_level = st.slider(
                    "Noise level (σ)",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1
                )

                candidate_models = st.multiselect(
                    "Candidate models to consider:",
                    ["Constant", "Linear", "Quadratic", "Cubic", "Multiple regression"],
                    default=["Constant", "Linear", "Quadratic"]
                )

                n_ic_simulations = st.slider(
                    "Number of simulations",
                    min_value=100,
                    max_value=1000,
                    value=500,
                    step=100
                )

                # Generate sample data for visualization
                np.random.seed(42)
                n_sample = 50

                if true_order != "multi":
                    x_sample = np.linspace(-3, 3, n_sample)

                    if true_order == 1:
                        y_true = true_params[0] + true_params[1] * x_sample
                    else:  # Quadratic
                        y_true = true_params[0] + true_params[1] * x_sample + true_params[2] * x_sample ** 2

                    y_sample = y_true + np.random.normal(0, noise_level, n_sample)

                    # Plot sample data and true function
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(x_sample, y_sample, alpha=0.6, label="Sample data")
                    ax.plot(x_sample, y_true, 'r-', label="True function")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_title("Sample Data and True Function")
                    ax.legend()
                    st.pyplot(fig)

                else:  # Multiple regression
                    x1_sample = np.random.uniform(-3, 3, n_sample)
                    x2_sample = np.random.uniform(-3, 3, n_sample)

                    y_true = true_params[0] + true_params[1] * x1_sample + true_params[2] * x2_sample
                    y_sample = y_true + np.random.normal(0, noise_level, n_sample)

                    # Create 3D plot
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x1_sample, x2_sample, y_sample, alpha=0.6)

                    # Create a meshgrid for the plane
                    x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
                    y_grid = true_params[0] + true_params[1] * x1_grid + true_params[2] * x2_grid

                    # Plot the true plane
                    ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3, color='r')

                    ax.set_xlabel('x₁')
                    ax.set_ylabel('x₂')
                    ax.set_zlabel('y')
                    ax.set_title("Sample Data and True Function (3D)")
                    st.pyplot(fig)

            with col2:
                # Sample sizes to analyze
                sample_sizes = [10, 20, 30, 50, 100, 200]

                # Initialize results dictionary
                model_selection_results = {
                    "AIC": {size: {model: 0 for model in candidate_models} for size in sample_sizes},
                    "BIC": {size: {model: 0 for model in candidate_models} for size in sample_sizes}
                }

                # Run simulations
                for n in sample_sizes:
                    for _ in range(n_ic_simulations):
                        # Generate data
                        if true_order != "multi":
                            x = np.random.uniform(-3, 3, n)
                            x_powers = np.column_stack([x ** i for i in range(1, 4)])

                            if true_order == 1:
                                y_true = true_params[0] + true_params[1] * x
                            else:  # Quadratic
                                y_true = true_params[0] + true_params[1] * x + true_params[2] * x ** 2

                            y = y_true + np.random.normal(0, noise_level, n)

                            # Fit candidate models
                            model_fits = {}
                            model_aic = {}
                            model_bic = {}

                            if "Constant" in candidate_models:
                                X_const = np.ones((n, 1))
                                model_const = sm.OLS(y, X_const).fit()
                                model_fits["Constant"] = model_const
                                model_aic["Constant"] = model_const.aic
                                model_bic["Constant"] = model_const.bic

                            if "Linear" in candidate_models:
                                X_linear = np.column_stack((np.ones(n), x))
                                model_linear = sm.OLS(y, X_linear).fit()
                                model_fits["Linear"] = model_linear
                                model_aic["Linear"] = model_linear.aic
                                model_bic["Linear"] = model_linear.bic

                            if "Quadratic" in candidate_models:
                                X_quad = np.column_stack((np.ones(n), x, x ** 2))
                                model_quad = sm.OLS(y, X_quad).fit()
                                model_fits["Quadratic"] = model_quad
                                model_aic["Quadratic"] = model_quad.aic
                                model_bic["Quadratic"] = model_quad.bic

                            if "Cubic" in candidate_models:
                                X_cubic = np.column_stack((np.ones(n), x, x ** 2, x ** 3))
                                model_cubic = sm.OLS(y, X_cubic).fit()
                                model_fits["Cubic"] = model_cubic
                                model_aic["Cubic"] = model_cubic.aic
                                model_bic["Cubic"] = model_cubic.bic

                            if "Multiple regression" in candidate_models:
                                # Generate random variable (not in true model)
                                x2 = np.random.uniform(-3, 3, n)
                                X_multi = np.column_stack((np.ones(n), x, x2))
                                model_multi = sm.OLS(y, X_multi).fit()
                                model_fits["Multiple regression"] = model_multi
                                model_aic["Multiple regression"] = model_multi.aic
                                model_bic["Multiple regression"] = model_multi.bic

                        else:  # Multiple regression
                            x1 = np.random.uniform(-3, 3, n)
                            x2 = np.random.uniform(-3, 3, n)

                            y_true = true_params[0] + true_params[1] * x1 + true_params[2] * x2
                            y = y_true + np.random.normal(0, noise_level, n)

                            # Fit candidate models
                            model_fits = {}
                            model_aic = {}
                            model_bic = {}

                            if "Constant" in candidate_models:
                                X_const = np.ones((n, 1))
                                model_const = sm.OLS(y, X_const).fit()
                                model_fits["Constant"] = model_const
                                model_aic["Constant"] = model_const.aic
                                model_bic["Constant"] = model_const.bic

                            if "Linear" in candidate_models:
                                X_linear = np.column_stack((np.ones(n), x1))
                                model_linear = sm.OLS(y, X_linear).fit()
                                model_fits["Linear"] = model_linear
                                model_aic["Linear"] = model_linear.aic
                                model_bic["Linear"] = model_linear.bic

                            if "Quadratic" in candidate_models:
                                X_quad = np.column_stack((np.ones(n), x1, x1 ** 2))
                                model_quad = sm.OLS(y, X_quad).fit()
                                model_fits["Quadratic"] = model_quad
                                model_aic["Quadratic"] = model_quad.aic
                                model_bic["Quadratic"] = model_quad.bic

                            if "Multiple regression" in candidate_models:
                                X_multi = np.column_stack((np.ones(n), x1, x2))
                                model_multi = sm.OLS(y, X_multi).fit()
                                model_fits["Multiple regression"] = model_multi
                                model_aic["Multiple regression"] = model_multi.aic
                                model_bic["Multiple regression"] = model_multi.bic

                            if "Cubic" in candidate_models:
                                X_cubic = np.column_stack((np.ones(n), x1, x1 ** 2, x1 ** 3))
                                model_cubic = sm.OLS(y, X_cubic).fit()
                                model_fits["Cubic"] = model_cubic
                                model_aic["Cubic"] = model_cubic.aic
                                model_bic["Cubic"] = model_cubic.bic

                        # Select models with minimum AIC and BIC
                        if model_aic:  # Check if dictionary is not empty
                            best_aic_model = min(model_aic, key=model_aic.get)
                            model_selection_results["AIC"][n][best_aic_model] += 1

                        if model_bic:  # Check if dictionary is not empty
                            best_bic_model = min(model_bic, key=model_bic.get)
                            model_selection_results["BIC"][n][best_bic_model] += 1

                # Calculate selection frequencies
                selection_freq_aic = {
                    size: {model: count / n_ic_simulations for model, count in models.items()}
                    for size, models in model_selection_results["AIC"].items()
                }

                selection_freq_bic = {
                    size: {model: count / n_ic_simulations for model, count in models.items()}
                    for size, models in model_selection_results["BIC"].items()
                }

                # Identify the true model name
                if true_order == 1:
                    true_model_name = "Linear"
                elif true_order == 2:
                    true_model_name = "Quadratic"
                else:  # Multiple regression
                    true_model_name = "Multiple regression"

                # Create selection frequency plots
                st.subheader("Model Selection Frequency by Sample Size")

                # Prepare data for plotting
                aic_data = []
                for size in sample_sizes:
                    for model in candidate_models:
                        aic_data.append({
                            "Sample Size": size,
                            "Model": model,
                            "Selection Frequency": selection_freq_aic[size][model],
                            "Criterion": "AIC"
                        })

                bic_data = []
                for size in sample_sizes:
                    for model in candidate_models:
                        bic_data.append({
                            "Sample Size": size,
                            "Model": model,
                            "Selection Frequency": selection_freq_bic[size][model],
                            "Criterion": "BIC"
                        })

                all_data = pd.DataFrame(aic_data + bic_data)

                # Plot for AIC
                fig1, ax1 = plt.subplots(figsize=(10, 6))

                for model in candidate_models:
                    model_data = [selection_freq_aic[size][model] for size in sample_sizes]
                    linestyle = '-' if model == true_model_name else '--'
                    linewidth = 2.5 if model == true_model_name else 1.5
                    ax1.plot(sample_sizes, model_data, linestyle=linestyle, linewidth=linewidth, marker='o',
                             label=model)

                ax1.set_xlabel("Sample Size (n)")
                ax1.set_ylabel("Selection Frequency")
                ax1.set_title("Model Selection Frequency with AIC")
                ax1.grid(True, alpha=0.3)
                ax1.legend(title="Models", loc='best')

                if true_model_name in candidate_models:
                    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

                st.pyplot(fig1)

                # Plot for BIC
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                for model in candidate_models:
                    model_data = [selection_freq_bic[size][model] for size in sample_sizes]
                    linestyle = '-' if model == true_model_name else '--'
                    linewidth = 2.5 if model == true_model_name else 1.5
                    ax2.plot(sample_sizes, model_data, linestyle=linestyle, linewidth=linewidth, marker='o',
                             label=model)

                ax2.set_xlabel("Sample Size (n)")
                ax2.set_ylabel("Selection Frequency")
                ax2.set_title("Model Selection Frequency with BIC")
                ax2.grid(True, alpha=0.3)
                ax2.legend(title="Models", loc='best')

                if true_model_name in candidate_models:
                    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

                st.pyplot(fig2)

                # Add interpretations
                st.markdown("""
                            ### Key Observations:

                            1. **AIC vs BIC Behavior**: AIC tends to select more complex models than BIC, especially in small samples.
                               This is because BIC has a stronger penalty for model complexity.

                            2. **Convergence to True Model**: As sample size increases, both criteria become more likely to select 
                               the true model, but BIC typically converges faster due to its consistency property.

                            3. **Overfitting Risk**: With small samples, both criteria (especially AIC) have a non-negligible 
                               probability of selecting overly complex models.

                            4. **Underfitting Risk**: BIC may select underfitted models in small samples when the true model 
                               is relatively complex.

                            5. **Signal-to-Noise Ratio**: Higher noise levels make model selection more difficult, requiring 
                               larger samples for reliable selection.
                            """)

                # Numerical results table
                st.subheader("Selection Frequencies for the True Model")

                if true_model_name in candidate_models:
                    true_model_results = pd.DataFrame({
                        "Sample Size": sample_sizes,
                        "AIC Selection Frequency": [selection_freq_aic[size][true_model_name] for size in sample_sizes],
                        "BIC Selection Frequency": [selection_freq_bic[size][true_model_name] for size in sample_sizes]
                    })
                    st.table(true_model_results)
                else:
                    st.info("True model not included in candidate models.")

                with tab2:
                    st.subheader("Cross-Validation in Small Samples")

            st.markdown("""
                        Cross-validation is a popular technique for model selection and evaluation, but it faces 
                        unique challenges in small samples. This tool explores the performance of different
                        cross-validation strategies with limited data.
                        """)

            col1, col2 = st.columns([1, 2])

            with col1:
                cv_scenario = st.radio(
                    "Modeling scenario:",
                    ["Polynomial regression", "Variable selection"]
                )

                if cv_scenario == "Polynomial regression":
                    true_degree = st.radio(
                        "True polynomial degree:",
                        [1, 2, 3],
                        index=1
                    )

                    max_degree = st.slider(
                        "Maximum degree to consider",
                        min_value=1,
                        max_value=6,
                        value=4,
                        step=1
                    )

                else:  # Variable selection
                    true_variables = st.multiselect(
                        "True relevant variables:",
                        ["x₁", "x₂", "x₃", "x₄", "x₅"],
                        default=["x₁", "x₂"]
                    )

                    candidate_variables = st.multiselect(
                        "Candidate variables to consider:",
                        ["x₁", "x₂", "x₃", "x₄", "x₅", "x₆", "x₇", "x₈"],
                        default=["x₁", "x₂", "x₃", "x₄", "x₅"]
                    )

                    # Ensure true variables are in candidate set
                    for var in true_variables:
                        if var not in candidate_variables:
                            st.warning(f"Added {var} to candidate variables as it's in the true model.")
                            candidate_variables.append(var)

                cv_methods = st.multiselect(
                    "Cross-validation methods to compare:",
                    ["Hold-out (70/30)", "k-fold (k=5)", "k-fold (k=10)", "Leave-one-out (LOOCV)"],
                    default=["Hold-out (70/30)", "k-fold (k=5)", "Leave-one-out (LOOCV)"]
                )

                cv_noise_level = st.slider(
                    "Noise level (σ)",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    key="cv_noise"
                )

                n_cv_simulations = st.slider(
                    "Number of simulations",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50
                )

                # Generate and visualize sample data
                np.random.seed(42)
                n_sample = 50

                if cv_scenario == "Polynomial regression":
                    x_sample = np.linspace(-3, 3, n_sample)

                    # Generate true function
                    y_true = np.zeros_like(x_sample)
                    for degree in range(true_degree + 1):
                        # Generate reasonable coefficients
                        coef = 1.0 / (degree + 1) if degree > 0 else 1.0
                        y_true += coef * x_sample ** degree

                    y_sample = y_true + np.random.normal(0, cv_noise_level, n_sample)

                    # Plot sample data and true function
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.scatter(x_sample, y_sample, alpha=0.6, label="Sample data")
                    ax.plot(x_sample, y_true, 'r-', label="True function")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_title(f"Sample Data and True {true_degree}-degree Polynomial")
                    ax.legend()
                    st.pyplot(fig)

                else:  # Variable selection
                    # Map variable names to indices
                    var_indices = {f"x₁": 0, f"x₂": 1, f"x₃": 2, f"x₄": 3,
                                   f"x₅": 4, f"x₆": 5, f"x₇": 6, f"x₈": 7}

                    # Generate features
                    X_sample = np.random.normal(0, 1, (n_sample, 8))

                    # Generate response
                    y_true = np.zeros(n_sample)
                    for var in true_variables:
                        idx = var_indices[var]
                        y_true += (1.0 / (idx + 1)) * X_sample[:, idx]  # Decreasing importance

                    y_sample = y_true + np.random.normal(0, cv_noise_level, n_sample)

                    # Create correlation heatmap
                    corr_data = np.corrcoef(np.column_stack((X_sample, y_sample.reshape(-1, 1))).T)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm",
                                xticklabels=list(var_indices.keys()) + ["y"],
                                yticklabels=list(var_indices.keys()) + ["y"],
                                ax=ax)
                    ax.set_title("Correlation Matrix")
                    st.pyplot(fig)

            with col2:
                # Sample sizes to analyze
                if cv_scenario == "Polynomial regression":
                    sample_sizes = [10, 20, 30, 50, 100, 200]
                else:  # More data needed for variable selection
                    sample_sizes = [20, 30, 50, 100, 200, 300]

                # Initialize results dictionary
                cv_results = {
                    method: {
                        size: {"correct": 0, "complexity": 0, "mse": 0}
                        for size in sample_sizes
                    }
                    for method in cv_methods
                }

                # Run simulations
                for n in sample_sizes:
                    for _ in range(n_cv_simulations):
                        # Generate data
                        if cv_scenario == "Polynomial regression":
                            x = np.random.uniform(-3, 3, n)

                            # Generate true function
                            y_true = np.zeros_like(x)
                            for degree in range(true_degree + 1):
                                coef = 1.0 / (degree + 1) if degree > 0 else 1.0
                                y_true += coef * x ** degree

                            y = y_true + np.random.normal(0, cv_noise_level, n)

                            # Define model candidates
                            models = []
                            for degree in range(1, max_degree + 1):
                                model = np.poly1d(np.polyfit(x, y, degree))
                                models.append(model)

                            # Apply CV methods
                            for method in cv_methods:
                                if method == "Hold-out (70/30)":
                                    # Split data
                                    split_idx = int(0.7 * n)
                                    if split_idx <= 1:  # Not enough data
                                        continue

                                    train_indices = np.random.choice(range(n), split_idx, replace=False)
                                    test_indices = np.array([i for i in range(n) if i not in train_indices])

                                    if len(test_indices) == 0:  # No test data
                                        continue

                                    x_train, x_test = x[train_indices], x[test_indices]
                                    y_train, y_test = y[train_indices], y[test_indices]

                                    # Evaluate models
                                    cv_errors = []
                                    for degree in range(1, max_degree + 1):
                                        model = np.poly1d(np.polyfit(x_train, y_train, degree))
                                        mse = np.mean((model(x_test) - y_test) ** 2)
                                        cv_errors.append(mse)

                                    best_degree = np.argmin(cv_errors) + 1

                                elif method.startswith("k-fold"):
                                    k = int(method.split("=")[1].split(")")[0])

                                    if n < k:  # Not enough data
                                        continue

                                    # Create folds
                                    indices = np.random.permutation(n)
                                    fold_size = n // k

                                    cv_errors_by_degree = [[] for _ in range(max_degree)]

                                    for fold in range(k):
                                        start_idx = fold * fold_size
                                        end_idx = start_idx + fold_size if fold < k - 1 else n

                                        test_indices = indices[start_idx:end_idx]
                                        train_indices = np.array([i for i in indices if i not in test_indices])

                                        if len(train_indices) == 0 or len(test_indices) == 0:
                                            continue

                                        x_train, x_test = x[train_indices], x[test_indices]
                                        y_train, y_test = y[train_indices], y[test_indices]

                                        # Evaluate models
                                        for degree in range(1, max_degree + 1):
                                            model = np.poly1d(np.polyfit(x_train, y_train, degree))
                                            mse = np.mean((model(x_test) - y_test) ** 2)
                                            cv_errors_by_degree[degree - 1].append(mse)

                                    # Average CV errors across folds
                                    mean_cv_errors = [np.mean(errors) if errors else float('inf')
                                                      for errors in cv_errors_by_degree]

                                    best_degree = np.argmin(mean_cv_errors) + 1

                                elif method == "Leave-one-out (LOOCV)":
                                    cv_errors_by_degree = []

                                    for degree in range(1, max_degree + 1):
                                        loo_errors = []

                                        for i in range(n):
                                            x_train = np.delete(x, i)
                                            y_train = np.delete(y, i)

                                            model = np.poly1d(np.polyfit(x_train, y_train, degree))
                                            error = (model(x[i]) - y[i]) ** 2
                                            loo_errors.append(error)

                                        cv_errors_by_degree.append(np.mean(loo_errors))

                                    best_degree = np.argmin(cv_errors_by_degree) + 1

                                # Record results
                                cv_results[method][n]["correct"] += (best_degree == true_degree)
                                cv_results[method][n]["complexity"] += best_degree

                                # Calculate out-of-sample MSE for the selected model
                                selected_model = np.poly1d(np.polyfit(x, y, best_degree))

                                # Generate new test data
                                x_new = np.random.uniform(-3, 3, 100)
                                y_true_new = np.zeros_like(x_new)
                                for degree in range(true_degree + 1):
                                    coef = 1.0 / (degree + 1) if degree > 0 else 1.0
                                    y_true_new += coef * x_new ** degree

                                test_mse = np.mean((selected_model(x_new) - y_true_new) ** 2)
                                cv_results[method][n]["mse"] += test_mse

                        else:  # Variable selection scenario
                            # Map variable names to indices
                            var_indices = {f"x₁": 0, f"x₂": 1, f"x₃": 2, f"x₄": 3,
                                           f"x₅": 4, f"x₆": 5, f"x₇": 6, f"x₈": 7}

                            # Generate features
                            X = np.random.normal(0, 1, (n, 8))

                            # Generate response
                            y_true = np.zeros(n)
                            for var in true_variables:
                                idx = var_indices[var]
                                y_true += (1.0 / (idx + 1)) * X[:, idx]

                            y = y_true + np.random.normal(0, cv_noise_level, n)

                            # Get indices of candidate variables
                            candidate_indices = [var_indices[var] for var in candidate_variables]
                            X_candidates = X[:, candidate_indices]

                            # Get indices of true variables among candidates
                            true_indices_in_candidates = [candidate_variables.index(var) for var in true_variables if
                                                          var in candidate_variables]

                            # Apply CV methods to each subset of variables
                            # (To simplify, we'll consider only subsets of sizes 1 to 5)
                            max_subset_size = min(5, len(candidate_variables))

                            for method in cv_methods:
                                best_subset = []
                                best_cv_error = float('inf')

                                for subset_size in range(1, max_subset_size + 1):
                                    # Consider subsets based on correlation with y
                                    corr_with_y = np.array([abs(np.corrcoef(X_candidates[:, j], y)[0, 1]) for j in
                                                            range(X_candidates.shape[1])])
                                    top_vars = np.argsort(corr_with_y)[::-1][:subset_size]

                                    if method == "Hold-out (70/30)":
                                        # Split data
                                        split_idx = int(0.7 * n)
                                        if split_idx <= 1:  # Not enough data
                                            continue

                                        train_indices = np.random.choice(range(n), split_idx, replace=False)
                                        test_indices = np.array([i for i in range(n) if i not in train_indices])

                                        X_train, X_test = X_candidates[train_indices], X_candidates[test_indices]
                                        y_train, y_test = y[train_indices], y[test_indices]

                                        # Fit model with this subset
                                        X_subset = X_train[:, top_vars]
                                        model = sm.OLS(y_train, sm.add_constant(X_subset)).fit()

                                        # Evaluate on test set
                                        X_test_subset = X_test[:, top_vars]
                                        y_pred = model.predict(sm.add_constant(X_test_subset))
                                        cv_error = np.mean((y_test - y_pred) ** 2)

                                    elif method.startswith("k-fold"):
                                        k = int(method.split("=")[1].split(")")[0])

                                        if n < k:  # Not enough data
                                            continue

                                        # Create folds
                                        indices = np.random.permutation(n)
                                        fold_size = n // k

                                        fold_errors = []

                                        for fold in range(k):
                                            start_idx = fold * fold_size
                                            end_idx = start_idx + fold_size if fold < k - 1 else n

                                            test_indices = indices[start_idx:end_idx]
                                            train_indices = np.array([i for i in indices if i not in test_indices])

                                            if len(train_indices) == 0 or len(test_indices) == 0:
                                                continue

                                            X_train, X_test = X_candidates[train_indices], X_candidates[test_indices]
                                            y_train, y_test = y[train_indices], y[test_indices]

                                            # Fit model with this subset
                                            X_subset = X_train[:, top_vars]
                                            try:
                                                model = sm.OLS(y_train, sm.add_constant(X_subset)).fit()

                                                # Evaluate on test set
                                                X_test_subset = X_test[:, top_vars]
                                                y_pred = model.predict(sm.add_constant(X_test_subset))
                                                fold_error = np.mean((y_test - y_pred) ** 2)
                                                fold_errors.append(fold_error)
                                            except:
                                                # Skip if model fitting fails
                                                continue

                                        cv_error = np.mean(fold_errors) if fold_errors else float('inf')

                                    elif method == "Leave-one-out (LOOCV)":
                                        loo_errors = []

                                        for i in range(n):
                                            X_train = np.delete(X_candidates, i, axis=0)
                                            y_train = np.delete(y, i)

                                            X_subset = X_train[:, top_vars]
                                            try:
                                                model = sm.OLS(y_train, sm.add_constant(X_subset)).fit()

                                                # Predict for left-out observation
                                                X_test = X_candidates[i, top_vars].reshape(1, -1)
                                                y_pred = model.predict(sm.add_constant(X_test))
                                                error = (y[i] - y_pred[0]) ** 2
                                                loo_errors.append(error)
                                            except:
                                                # Skip if model fitting fails
                                                continue

                                        cv_error = np.mean(loo_errors) if loo_errors else float('inf')

                                    # Update best subset if this performs better
                                    if cv_error < best_cv_error:
                                        best_cv_error = cv_error
                                        best_subset = top_vars.tolist()

                                # Record results
                                # Check if best subset includes all true variables
                                selected_vars = set([candidate_variables[idx] for idx in best_subset])
                                true_vars_set = set(true_variables)

                                # Count as correct if all true variables are selected
                                is_correct = true_vars_set.issubset(selected_vars)
                                cv_results[method][n]["correct"] += int(is_correct)

                                # Record complexity (number of variables)
                                cv_results[method][n]["complexity"] += len(best_subset)

                                # Calculate out-of-sample MSE for the selected model
                                # Generate new test data
                                X_new = np.random.normal(0, 1, (100, 8))

                                # Generate true response
                                y_true_new = np.zeros(100)
                                for var in true_variables:
                                    idx = var_indices[var]
                                    y_true_new += (1.0 / (idx + 1)) * X_new[:, idx]

                                # Fit model with selected variables on all training data
                                X_subset = X_candidates[:, best_subset]
                                try:
                                    model = sm.OLS(y, sm.add_constant(X_subset)).fit()

                                    # Evaluate on new test data
                                    X_new_candidates = X_new[:, candidate_indices]
                                    X_new_subset = X_new_candidates[:, best_subset]
                                    y_pred = model.predict(sm.add_constant(X_new_subset))

                                    test_mse = np.mean((y_true_new - y_pred) ** 2)
                                    cv_results[method][n]["mse"] += test_mse
                                except:
                                    # If model fitting fails, assign high MSE
                                    cv_results[method][n]["mse"] += 1000

                # Calculate average results
                for method in cv_methods:
                    for n in sample_sizes:
                        cv_results[method][n]["correct"] /= n_cv_simulations
                        cv_results[method][n]["complexity"] /= n_cv_simulations
                        cv_results[method][n]["mse"] /= n_cv_simulations

                # Create plots
                # Plot 1: Selection accuracy
                fig1, ax1 = plt.subplots(figsize=(10, 6))

                for method in cv_methods:
                    accuracy_values = [cv_results[method][n]["correct"] for n in sample_sizes]
                    ax1.plot(sample_sizes, accuracy_values, marker='o', linewidth=2, label=method)

                ax1.set_xlabel("Sample Size (n)")
                ax1.set_ylabel("Correct Model Selection Rate")

                if cv_scenario == "Polynomial regression":
                    ax1.set_title(f"Frequency of Selecting the True {true_degree}-degree Polynomial")
                else:
                    ax1.set_title("Frequency of Including All True Variables")

                ax1.grid(True, alpha=0.3)
                ax1.legend()

                st.pyplot(fig1)

                # Plot 2: Model complexity
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                for method in cv_methods:
                    complexity_values = [cv_results[method][n]["complexity"] for n in sample_sizes]
                    ax2.plot(sample_sizes, complexity_values, marker='o', linewidth=2, label=method)

                # Add reference line for true complexity
                if cv_scenario == "Polynomial regression":
                    true_complexity = true_degree
                    ax2.axhline(y=true_complexity, color='gray', linestyle='--', alpha=0.7,
                                label=f"True degree = {true_degree}")
                else:
                    true_complexity = len(true_variables)
                    ax2.axhline(y=true_complexity, color='gray', linestyle='--', alpha=0.7,
                                label=f"True # variables = {true_complexity}")

                ax2.set_xlabel("Sample Size (n)")

                if cv_scenario == "Polynomial regression":
                    ax2.set_ylabel("Average Selected Polynomial Degree")
                    ax2.set_title("Model Complexity: Selected Polynomial Degree")
                else:
                    ax2.set_ylabel("Average Number of Selected Variables")
                    ax2.set_title("Model Complexity: Number of Selected Variables")

                ax2.grid(True, alpha=0.3)
                ax2.legend()

                st.pyplot(fig2)

                # Plot 3: Out-of-sample MSE
                fig3, ax3 = plt.subplots(figsize=(10, 6))

                for method in cv_methods:
                    mse_values = [cv_results[method][n]["mse"] for n in sample_sizes]
                    ax3.plot(sample_sizes, mse_values, marker='o', linewidth=2, label=method)

                ax3.set_xlabel("Sample Size (n)")
                ax3.set_ylabel("Out-of-Sample MSE")
                ax3.set_title("Predictive Performance of Selected Models")
                ax3.grid(True, alpha=0.3)
                ax3.legend()

                # Use log scale if values vary widely
                if cv_scenario == "Variable selection":
                    ax3.set_yscale('log')

                st.pyplot(fig3)

                # Add interpretations
                st.markdown("""
                            ### Key Observations:

                            1. **Method Stability**: Different cross-validation strategies exhibit varying levels of stability 
                               in small samples. Leave-one-out CV tends to be more stable but can be computationally intensive.

                            2. **Complexity Bias**: With very small samples, some CV methods may exhibit bias toward 
                               simpler or more complex models, depending on the specific scenario.

                            3. **Sample Size Effects**: Performance of all CV methods improves with sample size, but at 
                               different rates. The relative advantages of different methods change with sample size.

                            4. **Overfitting Risk**: K-fold CV with small k may lead to overfitting in small samples, 
                               particularly when the number of observations per fold becomes very small.

                            5. **Variance of Estimates**: CV estimates of predictive performance have higher variance 
                               in small samples, making model selection less reliable.

                            6. **Signal-to-Noise Ratio**: Higher noise levels exacerbate small sample issues, requiring 
                               larger samples for reliable model selection.
                            """)

    ###########################################
    # PAGE: Regularization Solutions
    ###########################################
elif selected_page == "Regularization Solutions":
      st.title("Regularization Solutions for Small Samples")

      st.markdown("""
                    Regularization techniques are powerful tools for handling small sample challenges in econometrics.
                    This section demonstrates how different regularization approaches can improve estimation and prediction
                    in limited data scenarios.
                    """)

      tab1, tab2 = st.tabs(["Ridge & Lasso Regression", "Shrinkage Estimators"])

      with tab1:
        st.subheader("Ridge and Lasso Regression")

        st.markdown("""
                        Ridge and Lasso regression add penalty terms to the objective function to constrain coefficient
                        estimates, which can reduce overfitting in small samples. This tool explores their performance
                        across different sample sizes.
                        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            # Model configuration
            p_features = st.slider(
                "Number of features (p)",
                min_value=2,
                max_value=20,
                value=10,
                step=1
            )

            sparsity = st.slider(
                "True model sparsity (percentage of non-zero coefficients)",
                min_value=10,
                max_value=100,
                value=30,
                step=10
            )

            signal_strength = st.slider(
                "Signal strength",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )

            noise_level = st.slider(
                "Noise level (σ)",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key="reg_noise"
            )

            correlation = st.slider(
                "Feature correlation",
                min_value=0.0,
                max_value=0.9,
                value=0.3,
                step=0.1
            )

            regression_methods = st.multiselect(
                "Methods to compare:",
                ["OLS", "Ridge", "Lasso", "Elastic Net (α=0.5)"],
                default=["OLS", "Ridge", "Lasso"]
            )

            n_reg_simulations = st.slider(
                "Number of simulations",
                min_value=50,
                max_value=500,
                value=200,
                step=50
            )

            # Generate true beta coefficients
            np.random.seed(42)

            # Number of non-zero coefficients
            num_nonzero = max(1, int(p_features * sparsity / 100))

            # Generate true coefficients (some zero, some non-zero)
            true_beta = np.zeros(p_features)
            nonzero_indices = np.random.choice(p_features, num_nonzero, replace=False)
            true_beta[nonzero_indices] = signal_strength * np.random.choice([-1, 1], num_nonzero) * (
                        1 + np.random.rand(num_nonzero))

            # Show true coefficients
            st.subheader("True Coefficients")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.stem(range(p_features), true_beta)
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Coefficient Value")
            ax.set_title("True Model Coefficients")
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)


            # Generate correlated features
            def generate_correlated_features(n, p, correlation):
                # Create correlation matrix with specified off-diagonal correlation
                corr_matrix = np.ones((p, p)) * correlation
                np.fill_diagonal(corr_matrix, 1)

                # Generate multivariate normal data
                features = np.random.multivariate_normal(np.zeros(p), corr_matrix, n)
                return features


            # Generate sample data for visualization
            n_sample = 100
            X_sample = generate_correlated_features(n_sample, p_features, correlation)
            y_sample = X_sample @ true_beta + np.random.normal(0, noise_level, n_sample)

            # Show correlation matrix
            st.subheader("Feature Correlation Matrix")

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            corr_matrix = np.corrcoef(X_sample.T)
            sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax2)
            ax2.set_title(f"Feature Correlation Matrix (ρ = {correlation})")

            st.pyplot(fig2)

        with col2:
            # Sample sizes to analyze
            sample_sizes = [int(p_features * ratio) for ratio in [0.5, 0.75, 1, 1.5, 2, 3, 5, 10]]
            sample_sizes = sorted(list(set(sample_sizes)))  # Remove duplicates

            # Initialize results dictionary
            reg_results = {
                method: {
                    "n/p_ratio": [],
                    "mse_coef": [],
                    "mse_pred": [],
                    "variable_selection": [],
                    "sparsity": []
                }
                for method in regression_methods
            }

            # Run simulations
            for n in sample_sizes:
                n_p_ratio = n / p_features

                for _ in range(n_reg_simulations):
                    # Generate data
                    X_train = generate_correlated_features(n, p_features, correlation)
                    y_train = X_train @ true_beta + np.random.normal(0, noise_level, n)

                    # Generate test data
                    n_test = 1000
                    X_test = generate_correlated_features(n_test, p_features, correlation)
                    y_test = X_test @ true_beta + np.random.normal(0, noise_level, n_test)

                    # Standardize features
                    X_mean = X_train.mean(axis=0)
                    X_std = X_train.std(axis=0)

                    X_train_std = (X_train - X_mean) / X_std
                    X_test_std = (X_test - X_mean) / X_std

                    # Apply regression methods
                    for method in regression_methods:
                        if method == "OLS":
                            if n <= p_features:
                                # OLS not well-defined when n <= p, skip or use pseudoinverse
                                reg_results[method]["n/p_ratio"].append(n_p_ratio)
                                reg_results[method]["mse_coef"].append(float('inf'))
                                reg_results[method]["mse_pred"].append(float('inf'))
                                reg_results[method]["variable_selection"].append(0)
                                reg_results[method]["sparsity"].append(0)
                                continue

                            model = LinearRegression()
                            model.fit(X_train_std, y_train)
                            beta_hat = model.coef_

                        elif method == "Ridge":
                            # Find best alpha using cross-validation
                            alphas = np.logspace(-3, 3, 20)
                            model = Ridge(fit_intercept=True)

                            # Simple alpha selection - in practice, use CV
                            best_model = None
                            best_score = -float('inf')

                            for alpha in alphas:
                                model.set_params(alpha=alpha)
                                model.fit(X_train_std, y_train)
                                score = model.score(X_train_std, y_train)

                                if score > best_score:
                                    best_score = score
                                    best_model = model

                            beta_hat = best_model.coef_

                        elif method == "Lasso":
                            # Find best alpha using cross-validation
                            alphas = np.logspace(-3, 3, 20)
                            model = Lasso(fit_intercept=True, max_iter=10000)

                            # Simple alpha selection - in practice, use CV
                            best_model = None
                            best_score = -float('inf')

                            for alpha in alphas:
                                model.set_params(alpha=alpha)
                                try:
                                    model.fit(X_train_std, y_train)
                                    score = model.score(X_train_std, y_train)

                                    if score > best_score:
                                        best_score = score
                                        best_model = model
                                except:
                                    continue

                            if best_model is not None:
                                beta_hat = best_model.coef_
                            else:
                                # If all alpha values failed, use a simple ridge regression
                                model = Ridge(alpha=1.0)
                                model.fit(X_train_std, y_train)
                                beta_hat = model.coef_

                        elif method == "Elastic Net (α=0.5)":
                            # Find best alpha using cross-validation
                            alphas = np.logspace(-3, 3, 20)
                            model = ElasticNet(l1_ratio=0.5, fit_intercept=True, max_iter=10000)

                            # Simple alpha selection - in practice, use CV
                            best_model = None
                            best_score = -float('inf')

                            for alpha in alphas:
                                model.set_params(alpha=alpha)
                                try:
                                    model.fit(X_train_std, y_train)
                                    score = model.score(X_train_std, y_train)

                                    if score > best_score:
                                        best_score = score
                                        best_model = model
                                except:
                                    continue

                            if best_model is not None:
                                beta_hat = best_model.coef_
                            else:
                                # If all alpha values failed, use a simple ridge regression
                                model = Ridge(alpha=1.0)
                                model.fit(X_train_std, y_train)
                                beta_hat = model.coef_

                        # Evaluate results
                        # MSE of coefficient estimates
                        mse_coef = np.mean((beta_hat - true_beta) ** 2)

                        # Prediction MSE on test set
                        y_pred = X_test_std @ beta_hat
                        mse_pred = np.mean((y_test - y_pred) ** 2)

                        # Variable selection accuracy
                        # (percentage of correctly identified non-zero coefficients)
                        if np.count_nonzero(true_beta) > 0:
                            true_nonzero = true_beta != 0
                            pred_nonzero = np.abs(beta_hat) > 1e-4

                            # True positive rate (sensitivity)
                            tp = np.sum(true_nonzero & pred_nonzero)
                            actual_positives = np.sum(true_nonzero)
                            if actual_positives > 0:
                                sensitivity = tp / actual_positives
                            else:
                                sensitivity = 0

                            # False positive rate (1 - specificity)
                            fp = np.sum((~true_nonzero) & pred_nonzero)
                            actual_negatives = np.sum(~true_nonzero)
                            if actual_negatives > 0:
                                false_positive_rate = fp / actual_negatives
                            else:
                                false_positive_rate = 0

                            # Variable selection score (balanced accuracy)
                            variable_selection = (sensitivity + (1 - false_positive_rate)) / 2
                        else:
                            variable_selection = 0

                        # Sparsity (percentage of zero coefficients)
                        sparsity = np.mean(np.abs(beta_hat) < 1e-4) * 100

                        # Store results
                        reg_results[method]["n/p_ratio"].append(n_p_ratio)
                        reg_results[method]["mse_coef"].append(mse_coef)
                        reg_results[method]["mse_pred"].append(mse_pred)
                        reg_results[method]["variable_selection"].append(variable_selection)
                        reg_results[method]["sparsity"].append(sparsity)

            # Calculate average results for each sample size
            for method in regression_methods:
                avg_results = {
                    "n/p_ratio": [],
                    "mse_coef": [],
                    "mse_pred": [],
                    "variable_selection": [],
                    "sparsity": []
                }

                for n in sample_sizes:
                    n_p_ratio = n / p_features

                    # Find all results with this n/p ratio
                    indices = [i for i, ratio in enumerate(reg_results[method]["n/p_ratio"]) if ratio == n_p_ratio]

                    if indices:
                        avg_results["n/p_ratio"].append(n_p_ratio)
                        avg_results["mse_coef"].append(np.mean([reg_results[method]["mse_coef"][i] for i in indices]))
                        avg_results["mse_pred"].append(np.mean([reg_results[method]["mse_pred"][i] for i in indices]))
                        avg_results["variable_selection"].append(
                            np.mean([reg_results[method]["variable_selection"][i] for i in indices]))
                        avg_results["sparsity"].append(np.mean([reg_results[method]["sparsity"][i] for i in indices]))

                reg_results[method] = avg_results

            # Create plots
            # Plot 1: Coefficient MSE
            fig1, ax1 = plt.subplots(figsize=(10, 6))

            for method in regression_methods:
                n_p_ratios = reg_results[method]["n/p_ratio"]
                mse_coefs = reg_results[method]["mse_coef"]

                if "OLS" in method and any(np.isinf(mse_coefs)):
                    # Replace inf values with NaN for plotting
                    mse_coefs = [mse if not np.isinf(mse) else np.nan for mse in mse_coefs]

                ax1.plot(n_p_ratios, mse_coefs, marker='o', linewidth=2, label=method)

            ax1.set_xlabel("Sample Size to Features Ratio (n/p)")
            ax1.set_ylabel("Mean Squared Error (Coefficients)")
            ax1.set_title("Coefficient Estimation Error")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Use log scales if values vary widely
            ax1.set_xscale('log')
            ax1.set_yscale('log')

            st.pyplot(fig1)

            # Plot 2: Prediction MSE
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            for method in regression_methods:
                n_p_ratios = reg_results[method]["n/p_ratio"]
                mse_preds = reg_results[method]["mse_pred"]

                if "OLS" in method and any(np.isinf(mse_preds)):
                    # Replace inf values with NaN for plotting
                    mse_preds = [mse if not np.isinf(mse) else np.nan for mse in mse_preds]

                ax2.plot(n_p_ratios, mse_preds, marker='o', linewidth=2, label=method)

            ax2.set_xlabel("Sample Size to Features Ratio (n/p)")
            ax2.set_ylabel("Mean Squared Error (Prediction)")
            ax2.set_title("Out-of-Sample Prediction Error")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Use log scales if values vary widely
            ax2.set_xscale('log')
            ax2.set_yscale('log')

            st.pyplot(fig2)

            # Plot 3: Variable selection accuracy
            fig3, ax3 = plt.subplots(figsize=(10, 6))

            for method in regression_methods:
                n_p_ratios = reg_results[method]["n/p_ratio"]
                var_selection = reg_results[method]["variable_selection"]

                ax3.plot(n_p_ratios, var_selection, marker='o', linewidth=2, label=method)

            ax3.set_xlabel("Sample Size to Features Ratio (n/p)")
            ax3.set_ylabel("Variable Selection Accuracy")
            ax3.set_title("Accuracy in Identifying Relevant Variables")
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # Use log scale for x-axis if ratios vary widely
            ax3.set_xscale('log')

            st.pyplot(fig3)

            # Plot 4: Sparsity
            fig4, ax4 = plt.subplots(figsize=(10, 6))

            for method in regression_methods:
                n_p_ratios = reg_results[method]["n/p_ratio"]
                sparsity_vals = reg_results[method]["sparsity"]

                ax4.plot(n_p_ratios, sparsity_vals, marker='o', linewidth=2, label=method)

            # Add reference line for true sparsity
            true_sparsity = (1 - sparsity / 100) * 100
            ax4.axhline(y=true_sparsity, color='gray', linestyle='--', alpha=0.7,
                        label=f"True Sparsity ({true_sparsity:.1f}%)")

            ax4.set_xlabel("Sample Size to Features Ratio (n/p)")
            ax4.set_ylabel("Sparsity (% of zero coefficients)")
            ax4.set_title("Model Sparsity")
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            # Use log scale for x-axis if ratios vary widely
            ax4.set_xscale('log')

            st.pyplot(fig4)

            # Add interpretations
            st.markdown("""
                            ### Key Observations:

                            1. **Small Sample Performance**: Ridge and Lasso regularization significantly outperform OLS 
                               when the sample size is small relative to the number of features (n/p < 3).

                            2. **Prediction vs. Estimation**: Regularization methods often perform better on prediction tasks 
                               than on coefficient estimation, particularly in small samples.

                            3. **Variable Selection**: Lasso and Elastic Net can identify relevant variables even with limited 
                               data, with accuracy improving as sample size increases.

                            4. **Sparsity Recovery**: In sparse models, Lasso tends to recover the true model sparsity more 
                               accurately than Ridge as sample size increases.

                            5. **High Correlation Impact**: When features are highly correlated, Ridge regression often 
                               outperforms Lasso in terms of prediction error.

                            6. **n/p Threshold**: There's typically a critical n/p ratio (often around 2-3) where the benefits 
                               of regularization over OLS begin to diminish substantially.
                            """)

            # Add numerical results table
            st.subheader("Performance Comparison at n/p = 1")

            # Find results with n/p ratio closest to 1
            results_at_np1 = {}

            for method in regression_methods:
                n_p_ratios = reg_results[method]["n/p_ratio"]

                # Find the index closest to 1
                if n_p_ratios:
                    idx = np.argmin(np.abs(np.array(n_p_ratios) - 1))
                    results_at_np1[method] = {
                        "n/p Ratio": n_p_ratios[idx],
                        "Coef. MSE": reg_results[method]["mse_coef"][idx],
                        "Pred. MSE": reg_results[method]["mse_pred"][idx],
                        "Var. Selection": reg_results[method]["variable_selection"][idx],
                        "Sparsity (%)": reg_results[method]["sparsity"][idx]
                    }

            # Create table
            if results_at_np1:
                results_df = pd.DataFrame(results_at_np1).T
                st.table(results_df)
            else:
                st.info("No results available for n/p ratio near 1.")

            with tab2:
                st.subheader("Shrinkage Estimators")

        st.markdown("""
                        Shrinkage estimators like the James-Stein estimator can improve upon OLS in small samples
                        by shrinking coefficient estimates toward a prior value. This tool demonstrates the effectiveness
                        of several shrinkage approaches in reducing estimation error.
                        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            # Model configuration
            shrinkage_model = st.radio(
                "Model type:",
                ["Multiple means", "Linear regression"]
            )

            if shrinkage_model == "Multiple means":
                num_means = st.slider(
                    "Number of means (k)",
                    min_value=2,
                    max_value=50,
                    value=10,
                    step=1
                )

                mean_range = st.slider(
                    "Range of true means",
                    min_value=0.0,
                    max_value=10.0,
                    value=(0.0, 5.0),
                    step=0.5
                )

                noise_level = st.slider(
                    "Noise level (σ)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    key="shrink_noise"
                )

                shrink_target = st.radio(
                    "Shrinkage target:",
                    ["Grand mean", "Zero", "Custom value"]
                )

                if shrink_target == "Custom value":
                    custom_target = st.slider(
                        "Custom target value",
                        min_value=-5.0,
                        max_value=5.0,
                        value=0.0,
                        step=0.5
                    )

            else:  # Linear regression
                num_predictors = st.slider(
                    "Number of predictors (p)",
                    min_value=2,
                    max_value=20,
                    value=5,
                    step=1
                )

                coef_magnitude = st.slider(
                    "Coefficient magnitude",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1
                )

                noise_level = st.slider(
                    "Noise level (σ)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    key="shrink_reg_noise"
                )

                shrink_target = st.radio(
                    "Shrinkage target:",
                    ["Vector of zeros", "Custom vector"],
                    key="shrink_reg_target"
                )

                if shrink_target == "Custom vector":
                    st.info("Using random target vector with small values for demonstration")

            shrinkage_methods = st.multiselect(
                "Methods to compare:",
                ["MLE/OLS", "James-Stein", "Positive-part James-Stein", "Empirical Bayes"],
                default=["MLE/OLS", "James-Stein", "Positive-part James-Stein"]
            )

            n_shrink_simulations = st.slider(
                "Number of simulations",
                min_value=50,
                max_value=500,
                value=200,
                step=50
            )

            # Generate true parameters
            np.random.seed(42)

            if shrinkage_model == "Multiple means":
                # True means from specified range
                true_means = np.random.uniform(mean_range[0], mean_range[1], num_means)

                # Compute grand mean
                grand_mean = np.mean(true_means)

                # Set shrinkage target
                if shrink_target == "Grand mean":
                    target = grand_mean * np.ones(num_means)
                elif shrink_target == "Zero":
                    target = np.zeros(num_means)
                else:  # Custom value
                    target = custom_target * np.ones(num_means)

                # Show true means
                st.subheader("True Means")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.stem(range(num_means), true_means)
                ax.axhline(y=grand_mean, color='r', linestyle='--', label=f"Grand Mean ({grand_mean:.2f})")
                ax.axhline(y=target[0], color='g', linestyle='-.', label=f"Shrinkage Target ({target[0]:.2f})")
                ax.set_xlabel("Group Index")
                ax.set_ylabel("Mean Value")
                ax.set_title(f"{num_means} True Group Means")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

            else:  # Linear regression
                # True regression coefficients
                true_coefs = coef_magnitude * (np.random.rand(num_predictors) - 0.5)

                # Set shrinkage target
                if shrink_target == "Vector of zeros":
                    target = np.zeros(num_predictors)
                else:  # Custom vector
                    target = 0.1 * np.random.randn(num_predictors)

                # Show true coefficients
                st.subheader("True Regression Coefficients")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.stem(range(num_predictors), true_coefs)
                ax.plot(range(num_predictors), target, 'go-', linestyle='-.', label="Shrinkage Target")
                ax.set_xlabel("Coefficient Index")
                ax.set_ylabel("Coefficient Value")
                ax.set_title(f"{num_predictors} True Regression Coefficients")
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

        with col2:
            # Sample sizes to analyze
            if shrinkage_model == "Multiple means":
                param_count = num_means
                min_n = max(2, num_means // 10)
            else:  # Linear regression
                param_count = num_predictors
                min_n = num_predictors + 1

            sample_sizes = [min_n]
            while len(sample_sizes) < 7:
                next_n = int(sample_sizes[-1] * 2)
                if next_n > 500:
                    break
                sample_sizes.append(next_n)

            # Initialize results dictionary
            shrink_results = {
                method: {
                    "n": [],
                    "mse": [],
                    "risk_reduction": []  # Percentage risk reduction compared to MLE/OLS
                }
                for method in shrinkage_methods
            }

            # Run simulations
            for n in sample_sizes:
                method_mse = {method: [] for method in shrinkage_methods}

                for _ in range(n_shrink_simulations):
                    if shrinkage_model == "Multiple means":
                        # Generate data for multiple means model
                        X = np.zeros((n, num_means))

                        # Randomly assign observations to groups
                        group_assignments = np.random.choice(num_means, n)
                        for i, group in enumerate(group_assignments):
                            X[i, group] = 1

                        # Generate response
                        y = X @ true_means + np.random.normal(0, noise_level, n)

                        # Apply estimation methods
                        for method in shrinkage_methods:
                            if method == "MLE/OLS":
                                # MLE estimates for group means
                                theta_hat = np.zeros(num_means)
                                for j in range(num_means):
                                    group_obs = y[X[:, j] == 1]
                                    if len(group_obs) > 0:
                                        theta_hat[j] = np.mean(group_obs)
                                    else:
                                        # If no observations for this group, use grand mean
                                        theta_hat[j] = np.mean(y)

                            elif method == "James-Stein":
                                # First get MLE estimates
                                theta_mle = np.zeros(num_means)
                                n_j = np.zeros(num_means)  # observations per group
                                for j in range(num_means):
                                    group_obs = y[X[:, j] == 1]
                                    n_j[j] = len(group_obs)
                                    if n_j[j] > 0:
                                        theta_mle[j] = np.mean(group_obs)
                                    else:
                                        # If no observations for this group, use grand mean
                                        theta_mle[j] = np.mean(y)

                                # Calculate shrinkage factor
                                k = num_means
                                df = max(1, n - k)  # degrees of freedom
                                SSE = np.sum((y - X @ theta_mle) ** 2)
                                sigma2_hat = SSE / df

                                # Sum of squared deviations from target
                                S = np.sum((theta_mle - target) ** 2)

                                # James-Stein shrinkage factor
                                if S > 0:
                                    shrinkage = max(0, 1 - (k - 3) * sigma2_hat / S)
                                else:
                                    shrinkage = 0

                                # Apply shrinkage
                                theta_hat = target + (1 - shrinkage) * (theta_mle - target)

                            elif method == "Positive-part James-Stein":
                                # First get MLE estimates
                                theta_mle = np.zeros(num_means)
                                n_j = np.zeros(num_means)  # observations per group
                                for j in range(num_means):
                                    group_obs = y[X[:, j] == 1]
                                    n_j[j] = len(group_obs)
                                    if n_j[j] > 0:
                                        theta_mle[j] = np.mean(group_obs)
                                    else:
                                        # If no observations for this group, use grand mean
                                        theta_mle[j] = np.mean(y)

                                # Calculate shrinkage factor
                                k = num_means
                                df = max(1, n - k)  # degrees of freedom
                                SSE = np.sum((y - X @ theta_mle) ** 2)
                                sigma2_hat = SSE / df

                                # Sum of squared deviations from target
                                S = np.sum((theta_mle - target) ** 2)

                                # James-Stein shrinkage factor
                                if S > 0:
                                    shrinkage = max(0, 1 - (k - 3) * sigma2_hat / S)
                                else:
                                    shrinkage = 0

                                # Apply positive-part shrinkage
                                theta_hat = np.zeros(num_means)
                                for j in range(num_means):
                                    if np.abs(theta_mle[j] - target[j]) <= np.sqrt((k - 3) * sigma2_hat / S):
                                        theta_hat[j] = target[j]  # Complete shrinkage
                                    else:
                                        theta_hat[j] = target[j] + (1 - shrinkage) * (theta_mle[j] - target[j])

                            elif method == "Empirical Bayes":
                                # First get MLE estimates
                                theta_mle = np.zeros(num_means)
                                n_j = np.zeros(num_means)  # observations per group
                                var_j = np.zeros(num_means)  # variance for each group mean
                                for j in range(num_means):
                                    group_obs = y[X[:, j] == 1]
                                    n_j[j] = len(group_obs)
                                    if n_j[j] > 0:
                                        theta_mle[j] = np.mean(group_obs)
                                        var_j[j] = noise_level ** 2 / n_j[j]
                                    else:
                                        # If no observations for this group, use grand mean
                                        theta_mle[j] = np.mean(y)
                                        var_j[j] = noise_level ** 2

                                # Estimate prior variance
                                # Simple method: variance of MLE estimates minus average sampling variance
                                prior_var = max(0, np.var(theta_mle) - np.mean(var_j))

                                # Apply empirical Bayes shrinkage
                                theta_hat = np.zeros(num_means)
                                for j in range(num_means):
                                    if prior_var + var_j[j] > 0:
                                        weight = prior_var / (prior_var + var_j[j])
                                        # Weighted average of MLE and prior mean (target)
                                        theta_hat[j] = weight * theta_mle[j] + (1 - weight) * target[j]
                                    else:
                                        theta_hat[j] = target[j]

                        # Calculate MSE
                        mse = np.mean((theta_hat - true_means) ** 2)
                        method_mse[method].append(mse)

                    else:  # Linear regression
                        # Generate data for linear regression
                        X = np.random.normal(0, 1, (n, num_predictors))
                        y = X @ true_coefs + np.random.normal(0, noise_level, n)

                        # Apply estimation methods
                        for method in shrinkage_methods:
                            if method == "MLE/OLS":
                                # OLS estimates
                                if n > num_predictors:  # Only if well-defined
                                    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
                                else:
                                    # Use pseudoinverse for underdetermined system
                                    beta_hat = np.linalg.pinv(X) @ y

                            elif method == "James-Stein":
                                # First get OLS estimates
                                if n > num_predictors:  # Only if well-defined
                                    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
                                else:
                                    # Use pseudoinverse for underdetermined system
                                    beta_ols = np.linalg.pinv(X) @ y

                                # Calculate shrinkage factor
                                k = num_predictors
                                residuals = y - X @ beta_ols
                                SSE = np.sum(residuals ** 2)
                                df = max(1, n - k)  # degrees of freedom
                                sigma2_hat = SSE / df

                                # Sum of squared deviations from target
                                S = np.sum((beta_ols - target) ** 2)

                                # James-Stein shrinkage factor
                                if S > 0 and k > 2:
                                    shrinkage = max(0, 1 - (k - 2) * sigma2_hat / S)
                                else:
                                    shrinkage = 0

                                # Apply shrinkage
                                beta_hat = target + (1 - shrinkage) * (beta_ols - target)

                            elif method == "Positive-part James-Stein":
                                # First get OLS estimates
                                if n > num_predictors:  # Only if well-defined
                                    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
                                else:
                                    # Use pseudoinverse for underdetermined system
                                    beta_ols = np.linalg.pinv(X) @ y

                                # Calculate shrinkage factor
                                k = num_predictors
                                residuals = y - X @ beta_ols
                                SSE = np.sum(residuals ** 2)
                                df = max(1, n - k)  # degrees of freedom
                                sigma2_hat = SSE / df

                                # Sum of squared deviations from target
                                S = np.sum((beta_ols - target) ** 2)

                                # James-Stein shrinkage factor
                                if S > 0 and k > 2:
                                    shrinkage = max(0, 1 - (k - 2) * sigma2_hat / S)
                                else:
                                    shrinkage = 0

                                # Apply positive-part shrinkage
                                beta_hat = np.zeros(num_predictors)
                                for j in range(num_predictors):
                                    if k > 2 and np.abs(beta_ols[j] - target[j]) <= np.sqrt((k - 2) * sigma2_hat / S):
                                        beta_hat[j] = target[j]  # Complete shrinkage
                                    else:
                                        beta_hat[j] = target[j] + (1 - shrinkage) * (beta_ols[j] - target[j])

                            elif method == "Empirical Bayes":
                                # First get OLS estimates
                                if n > num_predictors:  # Only if well-defined
                                    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y

                                    # Calculate OLS covariance matrix
                                    residuals = y - X @ beta_ols
                                    SSE = np.sum(residuals ** 2)
                                    sigma2_hat = SSE / (n - num_predictors)
                                    cov_matrix = sigma2_hat * np.linalg.inv(X.T @ X)

                                    # Estimate prior variance (simplistic approach)
                                    # Variance of OLS estimates minus average diagonal of covariance
                                    var_ols = np.var(beta_ols)
                                    sampling_var = np.mean(np.diag(cov_matrix))
                                    prior_var = max(0, var_ols - sampling_var)

                                    # Apply empirical Bayes shrinkage
                                    beta_hat = np.zeros(num_predictors)
                                    for j in range(num_predictors):
                                        if prior_var + cov_matrix[j, j] > 0:
                                            weight = prior_var / (prior_var + cov_matrix[j, j])
                                            # Weighted average of OLS and prior mean (target)
                                            beta_hat[j] = weight * beta_ols[j] + (1 - weight) * target[j]
                                        else:
                                            beta_hat[j] = target[j]
                                else:
                                    # For underdetermined systems, fall back to ridge regression
                                    beta_ols = np.linalg.pinv(X) @ y

                                    # Apply simple shrinkage
                                    shrinkage = 0.5  # Fixed shrinkage for simplicity
                                    beta_hat = (1 - shrinkage) * beta_ols + shrinkage * target

                            # Calculate MSE
                            mse = np.mean((beta_hat - true_coefs) ** 2)
                            method_mse[method].append(mse)

                # Calculate average MSE for each method
                for method in shrinkage_methods:
                    avg_mse = np.mean(method_mse[method])
                    shrink_results[method]["n"].append(n)
                    shrink_results[method]["mse"].append(avg_mse)

                    # Calculate risk reduction compared to MLE/OLS
                    if "MLE/OLS" in shrinkage_methods:
                        ols_mse = np.mean(method_mse["MLE/OLS"])
                        if ols_mse > 0:
                            risk_reduction = (ols_mse - avg_mse) / ols_mse * 100
                        else:
                            risk_reduction = 0

                        shrink_results[method]["risk_reduction"].append(risk_reduction)
                    else:
                        shrink_results[method]["risk_reduction"].append(0)

            # Create plots
            # Plot 1: MSE vs Sample Size
            fig1, ax1 = plt.subplots(figsize=(10, 6))

            for method in shrinkage_methods:
                ax1.plot(shrink_results[method]["n"], shrink_results[method]["mse"],
                         marker='o', linewidth=2, label=method)

            ax1.set_xlabel("Sample Size (n)")
            ax1.set_ylabel("Mean Squared Error")

            if shrinkage_model == "Multiple means":
                ax1.set_title(f"Estimation Error for {num_means} Group Means")
            else:
                ax1.set_title(f"Estimation Error for {num_predictors} Regression Coefficients")

            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Use log scales if values vary widely
            ax1.set_xscale('log')
            ax1.set_yscale('log')

            st.pyplot(fig1)

            # Plot 2: Risk Reduction vs Sample Size
            if "MLE/OLS" in shrinkage_methods and len(shrinkage_methods) > 1:
                fig2, ax2 = plt.subplots(figsize=(10, 6))

                for method in shrinkage_methods:
                    if method != "MLE/OLS":
                        ax2.plot(shrink_results[method]["n"], shrink_results[method]["risk_reduction"],
                                 marker='o', linewidth=2, label=method)

                ax2.set_xlabel("Sample Size (n)")
                ax2.set_ylabel("Risk Reduction (%)")

                if shrinkage_model == "Multiple means":
                    ax2.set_title("Risk Reduction Compared to MLE")
                else:
                    ax2.set_title("Risk Reduction Compared to OLS")

                ax2.grid(True, alpha=0.3)
                ax2.legend()

                # Set x-axis to log scale
                ax2.set_xscale('log')

                # Add horizontal line at 0% (no improvement)
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

                st.pyplot(fig2)

            # Create animated visualization of shrinkage effect
            st.subheader("Visualization of Shrinkage Effect")

            if shrinkage_model == "Multiple means":
                # Generate sample data for visualization
                n_vis = 30

                # Randomly assign observations to groups
                X_vis = np.zeros((n_vis, num_means))
                group_assignments = np.random.choice(num_means, n_vis)
                for i, group in enumerate(group_assignments):
                    X_vis[i, group] = 1

                # Generate response
                y_vis = X_vis @ true_means + np.random.normal(0, noise_level, n_vis)

                # MLE estimates
                theta_mle = np.zeros(num_means)
                var_mle = np.zeros(num_means)
                for j in range(num_means):
                    group_obs = y_vis[X_vis[:, j] == 1]
                    if len(group_obs) > 0:
                        theta_mle[j] = np.mean(group_obs)
                        var_mle[j] = noise_level ** 2 / len(group_obs)
                    else:
                        # If no observations for this group, use grand mean
                        theta_mle[j] = np.mean(y_vis)
                        var_mle[j] = noise_level ** 2

                # James-Stein shrinkage
                k = num_means
                residuals = y_vis - X_vis @ theta_mle
                SSE = np.sum(residuals ** 2)
                df = max(1, n_vis - k)  # degrees of freedom
                sigma2_hat = SSE / df

                # Sum of squared deviations from target
                S = np.sum((theta_mle - target) ** 2)

                # James-Stein shrinkage factor
                if S > 0:
                    js_shrinkage = max(0, 1 - (k - 3) * sigma2_hat / S)
                else:
                    js_shrinkage = 0

                # Apply shrinkage with slider
                shrinkage_amount = st.slider(
                    "Shrinkage amount",
                    min_value=0.0,
                    max_value=1.0,
                    value=js_shrinkage,
                    step=0.01
                )

                # Apply selected shrinkage
                theta_shrunk = target + (1 - shrinkage_amount) * (theta_mle - target)

                # Plot estimates and true values
                fig3, ax3 = plt.subplots(figsize=(10, 6))

                # Plot true means
                ax3.scatter(range(num_means), true_means, s=100, color='blue', marker='o', label="True Means")

                # Plot MLE estimates with error bars
                ax3.errorbar(range(num_means), theta_mle, yerr=1.96 * np.sqrt(var_mle),
                             fmt='ro', label="MLE Estimates", alpha=0.7)

                # Plot shrunk estimates
                ax3.scatter(range(num_means), theta_shrunk, s=80, color='green', marker='*',
                            label=f"Shrunk Estimates (λ={shrinkage_amount:.2f})")

                # Plot shrinkage target
                ax3.axhline(y=target[0], color='gray', linestyle='--', label=f"Shrinkage Target")

                ax3.set_xlabel("Group Index")
                ax3.set_ylabel("Value")
                ax3.set_title("Effect of Shrinkage on Estimates")
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                st.pyplot(fig3)

                # Add MSE comparison
                mse_mle = np.mean((theta_mle - true_means) ** 2)
                mse_shrunk = np.mean((theta_shrunk - true_means) ** 2)

                improvement = (mse_mle - mse_shrunk) / mse_mle * 100 if mse_mle > 0 else 0

                st.info(f"""
                                **MSE Comparison:**
                                - MLE MSE: {mse_mle:.4f}
                                - Shrinkage MSE: {mse_shrunk:.4f}
                                - Improvement: {improvement:.1f}%
                                """)

            else:  # Linear regression visualization
                # Generate sample data for visualization
                n_vis = num_predictors + 5

                # Generate features
                X_vis = np.random.normal(0, 1, (n_vis, num_predictors))
                y_vis = X_vis @ true_coefs + np.random.normal(0, noise_level, n_vis)

                # OLS estimates
                if n_vis > num_predictors:
                    beta_ols = np.linalg.inv(X_vis.T @ X_vis) @ X_vis.T @ y_vis

                    # Calculate OLS standard errors
                    residuals = y_vis - X_vis @ beta_ols
                    SSE = np.sum(residuals ** 2)
                    sigma2_hat = SSE / (n_vis - num_predictors)
                    cov_matrix = sigma2_hat * np.linalg.inv(X_vis.T @ X_vis)
                    se_ols = np.sqrt(np.diag(cov_matrix))
                else:
                    beta_ols = np.linalg.pinv(X_vis) @ y_vis
                    se_ols = np.ones(num_predictors)  # Placeholder

                # Calculate James-Stein shrinkage factor
                k = num_predictors
                S = np.sum((beta_ols - target) ** 2)

                if S > 0 and k > 2:
                    js_shrinkage = max(0, 1 - (k - 2) * sigma2_hat / S)
                else:
                    js_shrinkage = 0.5  # Default value

                # Apply shrinkage with slider
                shrinkage_amount = st.slider(
                    "Shrinkage amount",
                    min_value=0.0,
                    max_value=1.0,
                    value=js_shrinkage,
                    step=0.01,
                    key="reg_shrink_amount"
                )

                # Apply selected shrinkage
                beta_shrunk = target + (1 - shrinkage_amount) * (beta_ols - target)

                # Plot estimates and true values
                fig3, ax3 = plt.subplots(figsize=(10, 6))

                # Plot true coefficients
                ax3.scatter(range(num_predictors), true_coefs, s=100, color='blue', marker='o',
                            label="True Coefficients")

                # Plot OLS estimates with error bars
                ax3.errorbar(range(num_predictors), beta_ols, yerr=1.96 * se_ols,
                             fmt='ro', label="OLS Estimates", alpha=0.7)

                # Plot shrunk estimates
                ax3.scatter(range(num_predictors), beta_shrunk, s=80, color='green', marker='*',
                            label=f"Shrunk Estimates (λ={shrinkage_amount:.2f})")

                # Plot shrinkage target
                ax3.plot(range(num_predictors), target, 'k--', label="Shrinkage Target")

                ax3.set_xlabel("Coefficient Index")
                ax3.set_ylabel("Value")
                ax3.set_title("Effect of Shrinkage on Coefficient Estimates")
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                st.pyplot(fig3)

                # Add MSE comparison
                mse_ols = np.mean((beta_ols - true_coefs) ** 2)
                mse_shrunk = np.mean((beta_shrunk - true_coefs) ** 2)

                improvement = (mse_ols - mse_shrunk) / mse_ols * 100 if mse_ols > 0 else 0

                st.info(f"""
                                **MSE Comparison:**
                                - OLS MSE: {mse_ols:.4f}
                                - Shrinkage MSE: {mse_shrunk:.4f}
                                - Improvement: {improvement:.1f}%
                                """)

            # Add interpretations
            st.markdown("""
                            ### Key Observations:

                            1. **Stein's Paradox**: James-Stein estimators can provide lower MSE than conventional 
                               estimators (MLE/OLS) when estimating multiple parameters simultaneously.

                            2. **Sample Size Effects**: The advantage of shrinkage estimators is most pronounced in 
                               small samples, and diminishes as sample size increases.

                            3. **Dimensionality**: The benefit of shrinkage increases with the number of parameters 
                               being estimated simultaneously (the dimensionality of the problem).

                            4. **Target Selection**: The choice of shrinkage target can significantly affect 
                               performance. An informed prior can enhance the benefits of shrinkage.

                            5. **Signal-to-Noise Ratio**: Shrinkage is more beneficial when the signal-to-noise ratio 
                               is low, as it effectively reduces the influence of noise on the estimates.

                            6. **Positive-Part Improvement**: The positive-part James-Stein estimator generally 
                               outperforms the original James-Stein estimator, especially when some parameters 
                               are very close to the target.
                            """)

###########################################
# PAGE: Bootstrap & Resampling
###########################################
elif selected_page == "Bootstrap & Resampling":
    st.title("Bootstrap & Resampling Methods")

    st.markdown("""
                    Bootstrap and other resampling techniques can help overcome small sample limitations
                    by providing robust inference without relying on asymptotic theory. This section explores
                    their effectiveness in various econometric scenarios.
                    """)

    tab1, tab2 = st.tabs(["Bootstrap Confidence Intervals", "Jackknife & Cross-Validation"])

    with tab1:
        st.subheader("Bootstrap Confidence Intervals")

        st.markdown("""
                        This tool demonstrates how bootstrap methods can provide more accurate confidence intervals
                        in small samples compared to traditional methods based on asymptotic theory.
                        """)

        col1, col2 = st.columns([1, 2])

        with col1:
            bootstrap_scenario = st.radio(
                "Statistic of interest:",
                ["Mean", "Median", "Correlation coefficient", "Regression coefficient"]
            )

            if bootstrap_scenario == "Mean":
                distribution = st.radio(
                    "Data distribution:",
                    ["Normal", "Log-normal", "t(3)", "Mixture"]
                )

                if distribution == "Mixture":
                    mix_ratio = st.slider(
                        "Mixture ratio (normal:outlier)",
                        min_value=0.5,
                        max_value=0.95,
                        value=0.8,
                        step=0.05
                    )

            elif bootstrap_scenario == "Median":
                distribution = st.radio(
                    "Data distribution:",
                    ["Normal", "Skewed", "Heavy-tailed", "Bimodal"]
                )

            elif bootstrap_scenario == "Correlation coefficient":
                true_correlation = st.slider(
                    "True correlation",
                    min_value=-0.9,
                    max_value=0.9,
                    value=0.5,
                    step=0.1
                )

                nonlinearity = st.slider(
                    "Nonlinearity strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )

            elif bootstrap_scenario == "Regression coefficient":
                coefficient_setting = st.radio(
                    "Regression setting:",
                    ["Simple linear", "Multiple with multicollinearity", "Heteroskedastic errors"]
                )

                if coefficient_setting == "Multiple with multicollinearity":
                    collinearity_strength = st.slider(
                        "Multicollinearity strength",
                        min_value=0.1,
                        max_value=0.95,
                        value=0.7,
                        step=0.05
                    )

                if coefficient_setting == "Heteroskedastic errors":
                    heteroskedasticity = st.slider(
                        "Heteroskedasticity strength",
                        min_value=0.1,
                        max_value=5.0,
                        value=2.0,
                        step=0.1
                    )

            sample_size = st.slider(
                "Sample size",
                min_value=10,
                max_value=200,
                value=30,
                step=5
            )

            bootstrap_reps = st.slider(
                "Bootstrap replications",
                min_value=500,
                max_value=5000,
                value=2000,
                step=500
            )

            run_bootstrap = st.button("Run Bootstrap Analysis")

        with col2:
            if run_bootstrap:
                st.markdown("### Bootstrap Results")

                # Generate synthetic data based on the selected scenario
                if bootstrap_scenario == "Mean":
                    if distribution == "Normal":
                        data = np.random.normal(loc=10, scale=5, size=sample_size)
                        true_param = 10
                    elif distribution == "Log-normal":
                        data = np.random.lognormal(mean=1.5, sigma=0.8, size=sample_size)
                        true_param = np.exp(1.5 + 0.8 ** 2 / 2)  # Mean of lognormal
                    elif distribution == "t(3)":
                        data = np.random.standard_t(df=3, size=sample_size) * 2 + 5
                        true_param = 5  # Mean of t-distribution with df > 1
                    elif distribution == "Mixture":
                        normal_samples = int(sample_size * mix_ratio)
                        outlier_samples = sample_size - normal_samples
                        data = np.concatenate([
                            np.random.normal(loc=10, scale=2, size=normal_samples),
                            np.random.normal(loc=10, scale=15, size=outlier_samples)
                        ])
                        true_param = 10

                    # Compute the sample statistic
                    sample_mean = np.mean(data)

                    # Asymptotic confidence interval
                    se = np.std(data, ddof=1) / np.sqrt(sample_size)
                    asymptotic_ci = (sample_mean - 1.96 * se, sample_mean + 1.96 * se)

                    # Bootstrap confidence interval
                    bootstrap_means = []
                    for _ in range(bootstrap_reps):
                        bootstrap_sample = np.random.choice(data, size=sample_size, replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))

                    bootstrap_means = np.array(bootstrap_means)
                    bootstrap_ci_percentile = np.percentile(bootstrap_means, [2.5, 97.5])

                    # Plotting
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                    # Original data with sample and true means
                    ax[0].hist(data, bins=20, alpha=0.7, color='skyblue', density=True)
                    ax[0].axvline(sample_mean, color='red', linestyle='-', label=f'Sample mean: {sample_mean:.2f}')
                    ax[0].axvline(true_param, color='green', linestyle='--', label=f'True mean: {true_param:.2f}')
                    ax[0].set_title(f'Sample Distribution (n={sample_size})')
                    ax[0].legend()

                    # Bootstrap distribution
                    ax[1].hist(bootstrap_means, bins=30, alpha=0.7, color='lightgreen', density=True)
                    ax[1].axvline(sample_mean, color='red', linestyle='-', label=f'Sample mean: {sample_mean:.2f}')
                    ax[1].axvline(bootstrap_ci_percentile[0], color='purple', linestyle='--',
                                  label=f'Bootstrap CI: [{bootstrap_ci_percentile[0]:.2f}, {bootstrap_ci_percentile[1]:.2f}]')
                    ax[1].axvline(bootstrap_ci_percentile[1], color='purple', linestyle='--')
                    ax[1].axvline(asymptotic_ci[0], color='orange', linestyle=':',
                                  label=f'Asymptotic CI: [{asymptotic_ci[0]:.2f}, {asymptotic_ci[1]:.2f}]')
                    ax[1].axvline(asymptotic_ci[1], color='orange', linestyle=':')
                    ax[1].set_title(f'Bootstrap Distribution ({bootstrap_reps} replications)')
                    ax[1].legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Coverage analysis
                    in_bootstrap_ci = (true_param >= bootstrap_ci_percentile[0]) and (
                                true_param <= bootstrap_ci_percentile[1])
                    in_asymptotic_ci = (true_param >= asymptotic_ci[0]) and (true_param <= asymptotic_ci[1])

                    st.markdown(f"""
                                **Coverage Analysis:**
                                - True parameter value: {true_param:.4f}
                                - Sample estimate: {sample_mean:.4f}
                                - Bootstrap CI width: {bootstrap_ci_percentile[1] - bootstrap_ci_percentile[0]:.4f}
                                - Asymptotic CI width: {asymptotic_ci[1] - asymptotic_ci[0]:.4f}
                                - True parameter in bootstrap CI: {'✅' if in_bootstrap_ci else '❌'}
                                - True parameter in asymptotic CI: {'✅' if in_asymptotic_ci else '❌'}
                                """)

                    # Provide download link for the figure
                    st.markdown(get_image_download_link(fig, "bootstrap_mean_analysis.png",
                                                        "Download Bootstrap Analysis Figure"), unsafe_allow_html=True)

                elif bootstrap_scenario == "Median":
                    if distribution == "Normal":
                        data = np.random.normal(loc=10, scale=5, size=sample_size)
                        true_param = 10
                    elif distribution == "Skewed":
                        data = np.random.lognormal(mean=1.5, sigma=0.8, size=sample_size)
                        true_param = np.exp(1.5)  # Median of lognormal
                    elif distribution == "Heavy-tailed":
                        data = np.random.standard_t(df=2.5, size=sample_size) * 2 + 5
                        true_param = 5  # Median of symmetric t-distribution
                    elif distribution == "Bimodal":
                        data = np.concatenate([
                            np.random.normal(loc=5, scale=1, size=sample_size // 2),
                            np.random.normal(loc=15, scale=1, size=sample_size - sample_size // 2)
                        ])
                        # For bimodal, true median depends on the exact mixture
                        true_param = np.median(np.concatenate([
                            np.random.normal(loc=5, scale=1, size=100000 // 2),
                            np.random.normal(loc=15, scale=1, size=100000 - 100000 // 2)
                        ]))

                    # Compute the sample statistic
                    sample_median = np.median(data)

                    # Bootstrap confidence interval
                    bootstrap_medians = []
                    for _ in range(bootstrap_reps):
                        bootstrap_sample = np.random.choice(data, size=sample_size, replace=True)
                        bootstrap_medians.append(np.median(bootstrap_sample))

                    bootstrap_medians = np.array(bootstrap_medians)
                    bootstrap_ci_percentile = np.percentile(bootstrap_medians, [2.5, 97.5])

                    # Calculate approximate asymptotic CI for median
                    # Based on order statistics approximation
                    density_est = stats.gaussian_kde(data)
                    median_idx = np.argsort(data)[sample_size // 2]
                    f_median_est = density_est(sample_median)[0]
                    asymptotic_se = 1.0 / (2 * f_median_est * np.sqrt(sample_size))
                    asymptotic_ci = (sample_median - 1.96 * asymptotic_se, sample_median + 1.96 * asymptotic_se)

                    # Plotting
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                    # Original data with sample and true medians
                    ax[0].hist(data, bins=20, alpha=0.7, color='skyblue', density=True)
                    ax[0].axvline(sample_median, color='red', linestyle='-',
                                  label=f'Sample median: {sample_median:.2f}')
                    ax[0].axvline(true_param, color='green', linestyle='--', label=f'True median: {true_param:.2f}')
                    ax[0].set_title(f'Sample Distribution (n={sample_size})')
                    ax[0].legend()

                    # Bootstrap distribution
                    ax[1].hist(bootstrap_medians, bins=30, alpha=0.7, color='lightgreen', density=True)
                    ax[1].axvline(sample_median, color='red', linestyle='-',
                                  label=f'Sample median: {sample_median:.2f}')
                    ax[1].axvline(bootstrap_ci_percentile[0], color='purple', linestyle='--',
                                  label=f'Bootstrap CI: [{bootstrap_ci_percentile[0]:.2f}, {bootstrap_ci_percentile[1]:.2f}]')
                    ax[1].axvline(bootstrap_ci_percentile[1], color='purple', linestyle='--')
                    ax[1].axvline(asymptotic_ci[0], color='orange', linestyle=':',
                                  label=f'Asymptotic CI: [{asymptotic_ci[0]:.2f}, {asymptotic_ci[1]:.2f}]')
                    ax[1].axvline(asymptotic_ci[1], color='orange', linestyle=':')
                    ax[1].set_title(f'Bootstrap Distribution ({bootstrap_reps} replications)')
                    ax[1].legend()

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Coverage analysis
                    in_bootstrap_ci = (true_param >= bootstrap_ci_percentile[0]) and (
                                true_param <= bootstrap_ci_percentile[1])
                    in_asymptotic_ci = (true_param >= asymptotic_ci[0]) and (true_param <= asymptotic_ci[1])

                    st.markdown(f"""
                                **Coverage Analysis:**
                                - True parameter value: {true_param:.4f}
                                - Sample estimate: {sample_median:.4f}
                                - Bootstrap CI width: {bootstrap_ci_percentile[1] - bootstrap_ci_percentile[0]:.4f}
                                - Asymptotic CI width: {asymptotic_ci[1] - asymptotic_ci[0]:.4f}
                                - True parameter in bootstrap CI: {'✅' if in_bootstrap_ci else '❌'}
                                - True parameter in asymptotic CI: {'✅' if in_asymptotic_ci else '❌'}
                                """)

                    # Provide download link for the figure
                    st.markdown(get_image_download_link(fig, "bootstrap_median_analysis.png",
                                                        "Download Bootstrap Analysis Figure"), unsafe_allow_html=True)

                elif bootstrap_scenario == "Correlation coefficient":
                    # Generate data with specified correlation
                    if nonlinearity == 0:
                        # Linear relationship
                        cov_matrix = [[1, true_correlation], [true_correlation, 1]]
                        data = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix, size=sample_size)
                        true_param = true_correlation
                    else:
                        # Nonlinear relationship with specified correlation
                        x = np.random.normal(0, 1, sample_size)
                        # Add nonlinear component
                        y_nonlinear = nonlinearity * (x ** 2 - 1)  # E[x^2] = 1 for standard normal
                        # Add linear component to achieve target correlation
                        rho_linear = np.sqrt(true_correlation ** 2 / (1 - nonlinearity ** 2)) if nonlinearity < 1 else 0
                        y_linear = rho_linear * x
                        y = y_linear + y_nonlinear + np.random.normal(0,
                                                                      np.sqrt(1 - rho_linear ** 2 - nonlinearity ** 2),
                                                                      sample_size)

                        data = np.column_stack((x, y))
                        true_param = true_correlation

                    # Compute the sample statistic
                    sample_corr = np.corrcoef(data.T)[0, 1]

                    # Fisher transformation for asymptotic CI
                    z = np.arctanh(sample_corr)
                    se_z = 1 / np.sqrt(sample_size - 3)
                    z_ci = (z - 1.96 * se_z, z + 1.96 * se_z)
                    asymptotic_ci = (np.tanh(z_ci[0]), np.tanh(z_ci[1]))

                    # Bootstrap confidence interval
                    bootstrap_corrs = []
                    for _ in range(bootstrap_reps):
                        indices = np.random.choice(sample_size, size=sample_size, replace=True)
                        bootstrap_sample = data[indices]
                        bootstrap_corrs.append(np.corrcoef(bootstrap_sample.T)[0, 1])

                    bootstrap_corrs = np.array(bootstrap_corrs)
                    bootstrap_ci_percentile = np.percentile(bootstrap_corrs, [2.5, 97.5])

                    # Plotting
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                    # Scatter plot of original data
                    ax[0].scatter(data[:, 0], data[:, 1], alpha=0.7, color='skyblue')
                    ax[0].set_title(f'Sample Data (n={sample_size}, r={sample_corr:.2f})')
                    ax[0].set_xlabel('X')
                    ax[0].set_ylabel('Y')

                    # Add correlation information
                    ax[0].text(0.05, 0.95, f"Sample correlation: {sample_corr:.4f}\nTrue correlation: {true_param:.4f}",
                               transform=ax[0].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                    # Bootstrap distribution
                    ax[1].hist(bootstrap_corrs, bins=30, alpha=0.7, color='lightgreen', density=True)
                    ax[1].axvline(sample_corr, color='red', linestyle='-', label=f'Sample r: {sample_corr:.4f}')
                    ax[1].axvline(true_param, color='green', linestyle='--', label=f'True r: {true_param:.4f}')
                    ax[1].axvline(bootstrap_ci_percentile[0], color='purple', linestyle='--',
                                  label=f'Bootstrap CI: [{bootstrap_ci_percentile[0]:.4f}, {bootstrap_ci_percentile[1]:.4f}]')
                    ax[1].axvline(bootstrap_ci_percentile[1], color='purple', linestyle='--')
                    ax[1].axvline(asymptotic_ci[0], color='orange', linestyle=':',
                                  label=f'Fisher CI: [{asymptotic_ci[0]:.4f}, {asymptotic_ci[1]:.4f}]')
                    ax[1].axvline(asymptotic_ci[1], color='orange', linestyle=':')
                    ax[1].set_title(f'Bootstrap Distribution ({bootstrap_reps} replications)')
                    ax[1].legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=2)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Coverage analysis
                    in_bootstrap_ci = (true_param >= bootstrap_ci_percentile[0]) and (
                                true_param <= bootstrap_ci_percentile[1])
                    in_asymptotic_ci = (true_param >= asymptotic_ci[0]) and (true_param <= asymptotic_ci[1])

                    st.markdown(f"""
                                **Coverage Analysis:**
                                - True correlation: {true_param:.4f}
                                - Sample correlation: {sample_corr:.4f}
                                - Bootstrap CI width: {bootstrap_ci_percentile[1] - bootstrap_ci_percentile[0]:.4f}
                                - Fisher CI width: {asymptotic_ci[1] - asymptotic_ci[0]:.4f}
                                - True parameter in bootstrap CI: {'✅' if in_bootstrap_ci else '❌'}
                                - True parameter in asymptotic CI: {'✅' if in_asymptotic_ci else '❌'}
                                """)

                    # Provide download link for the figure
                    st.markdown(get_image_download_link(fig, "bootstrap_correlation_analysis.png",
                                                        "Download Bootstrap Analysis Figure"), unsafe_allow_html=True)

                elif bootstrap_scenario == "Regression coefficient":
                    if coefficient_setting == "Simple linear":
                        # Simple linear regression with normal errors
                        X = np.random.normal(0, 1, sample_size)
                        beta0, beta1 = 1.5, 2.0  # True coefficients
                        Y = beta0 + beta1 * X + np.random.normal(0, 2, sample_size)
                        X_design = sm.add_constant(X.reshape(-1, 1))

                    elif coefficient_setting == "Multiple with multicollinearity":
                        # Multiple regression with multicollinearity
                        X1 = np.random.normal(0, 1, sample_size)
                        X2 = collinearity_strength * X1 + np.sqrt(1 - collinearity_strength ** 2) * np.random.normal(0,
                                                                                                                     1,
                                                                                                                     sample_size)
                        beta0, beta1, beta2 = 1.0, 2.0, -0.5  # True coefficients
                        Y = beta0 + beta1 * X1 + beta2 * X2 + np.random.normal(0, 1, sample_size)
                        X_design = sm.add_constant(np.column_stack((X1, X2)))
                        X = np.column_stack((X1, X2))

                    elif coefficient_setting == "Heteroskedastic errors":
                        # Regression with heteroskedastic errors
                        X = np.random.normal(0, 1, sample_size)
                        beta0, beta1 = 1.0, 2.0  # True coefficients
                        # Variance increases with X
                        Y = beta0 + beta1 * X + np.random.normal(0, 0.5 + heteroskedasticity * np.abs(X), sample_size)
                        X_design = sm.add_constant(X.reshape(-1, 1))

                    # OLS regression
                    model = sm.OLS(Y, X_design)
                    results = model.fit()

                    # Extract coefficient of interest
                    if coefficient_setting == "Simple linear":
                        coef_idx = 1  # Index for beta1
                        true_param = beta1
                        param_name = "β₁"
                    elif coefficient_setting == "Multiple with multicollinearity":
                        coef_idx = 1  # Index for beta1
                        true_param = beta1
                        param_name = "β₁"
                    else:
                        coef_idx = 1  # Index for beta1
                        true_param = beta1
                        param_name = "β₁"

                    sample_coef = results.params[coef_idx]
                    asymptotic_ci = results.conf_int(alpha=0.05)[coef_idx]

                    # Bootstrap confidence interval
                    bootstrap_coefs = []
                    for _ in range(bootstrap_reps):
                        if coefficient_setting == "Simple linear" or coefficient_setting == "Heteroskedastic errors":
                            # For simple regression, resample (x,y) pairs
                            indices = np.random.choice(sample_size, size=sample_size, replace=True)
                            X_boot = X[indices]
                            Y_boot = Y[indices]
                            X_design_boot = sm.add_constant(X_boot.reshape(-1, 1))
                        else:
                            # For multiple regression, resample (x1,x2,y) tuples
                            indices = np.random.choice(sample_size, size=sample_size, replace=True)
                            X_boot = X[indices]
                            Y_boot = Y[indices]
                            X_design_boot = sm.add_constant(X_boot)

                        model_boot = sm.OLS(Y_boot, X_design_boot)
                        results_boot = model_boot.fit()
                        bootstrap_coefs.append(results_boot.params[coef_idx])

                    bootstrap_coefs = np.array(bootstrap_coefs)
                    bootstrap_ci_percentile = np.percentile(bootstrap_coefs, [2.5, 97.5])

                    # Plotting
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                    # Data and regression line
                    if coefficient_setting == "Simple linear" or coefficient_setting == "Heteroskedastic errors":
                        ax[0].scatter(X, Y, alpha=0.6, color='skyblue')
                        x_range = np.linspace(X.min(), X.max(), 100)
                        ax[0].plot(x_range, results.params[0] + results.params[1] * x_range, 'r-',
                                   label=f'Sample: Y = {results.params[0]:.2f} + {results.params[1]:.2f}X')
                        ax[0].plot(x_range, beta0 + beta1 * x_range, 'g--',
                                   label=f'True: Y = {beta0:.2f} + {beta1:.2f}X')
                        ax[0].set_xlabel('X')
                        ax[0].set_ylabel('Y')
                        ax[0].set_title(f'Sample Data (n={sample_size})')
                        ax[0].legend()
                    else:
                        # For multiple regression, show partial regression plot
                        sm.graphics.plot_partregress(results, 'x1', 'y', ax=ax[0])
                        ax[0].set_title(f'Partial Regression Plot (n={sample_size})')

                    # Bootstrap distribution
                    ax[1].hist(bootstrap_coefs, bins=30, alpha=0.7, color='lightgreen', density=True)
                    ax[1].axvline(sample_coef, color='red', linestyle='-',
                                  label=f'Sample {param_name}: {sample_coef:.4f}')
                    ax[1].axvline(true_param, color='green', linestyle='--',
                                  label=f'True {param_name}: {true_param:.4f}')
                    ax[1].axvline(bootstrap_ci_percentile[0], color='purple', linestyle='--',
                                  label=f'Bootstrap CI: [{bootstrap_ci_percentile[0]:.4f}, {bootstrap_ci_percentile[1]:.4f}]')
                    ax[1].axvline(bootstrap_ci_percentile[1], color='purple', linestyle='--')
                    ax[1].axvline(asymptotic_ci[0], color='orange', linestyle=':',
                                  label=f'OLS CI: [{asymptotic_ci[0]:.4f}, {asymptotic_ci[1]:.4f}]')
                    ax[1].axvline(asymptotic_ci[1], color='orange', linestyle=':')
                    ax[1].set_title(f'Bootstrap Distribution ({bootstrap_reps} replications)')
                    ax[1].legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=2)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Coverage analysis
                    in_bootstrap_ci = (true_param >= bootstrap_ci_percentile[0]) and (
                                true_param <= bootstrap_ci_percentile[1])
                    in_asymptotic_ci = (true_param >= asymptotic_ci[0]) and (true_param <= asymptotic_ci[1])

                    st.markdown(f"""
                                **Coverage Analysis:**
                                - True {param_name}: {true_param:.4f}
                                - Sample {param_name}: {sample_coef:.4f}
                                - Bootstrap CI width: {bootstrap_ci_percentile[1] - bootstrap_ci_percentile[0]:.4f}
                                - OLS CI width: {asymptotic_ci[1] - asymptotic_ci[0]:.4f}
                                - True parameter in bootstrap CI: {'✅' if in_bootstrap_ci else '❌'}
                                - True parameter in asymptotic CI: {'✅' if in_asymptotic_ci else '❌'}

                                **Note:** Bootstrap methods are particularly valuable when standard assumptions are violated,
                                such as with non-normal errors, heteroskedasticity, or multicollinearity.
                                """)

                    # Provide download link for the figure
                    st.markdown(get_image_download_link(fig, "bootstrap_regression_analysis.png",
                                                        "Download Bootstrap Analysis Figure"), unsafe_allow_html=True)

        with tab2:
            st.subheader("Jackknife & Cross-Validation")

            st.markdown("""
                                The jackknife is another resampling technique that can be used to estimate bias and variance
                                of a statistic, while cross-validation helps assess model performance in small samples.
                                """)

            col1, col2 = st.columns([1, 2])

            with col1:
                resampling_method = st.radio(
                    "Method to demonstrate:",
                    ["Jackknife bias correction", "K-fold cross-validation"]
                )

                if resampling_method == "Jackknife bias correction":
                    statistic_type = st.radio(
                        "Statistic of interest:",
                        ["Variance estimate", "Correlation estimate", "R-squared estimate"]
                    )

                else:  # K-fold cross-validation
                    model_type = st.radio(
                        "Model type:",
                        ["Linear regression", "Ridge regression"]
                    )

                    k_folds = st.slider(
                        "Number of folds",
                        min_value=2,
                        max_value=10,
                        value=5,
                        step=1
                    )

                jackknife_sample_size = st.slider(
                    "Sample size",
                    min_value=10,
                    max_value=100,
                    value=20,
                    step=5
                )

                run_analysis = st.button("Run Analysis")

            with col2:
                if run_analysis:
                    if resampling_method == "Jackknife bias correction":
                        st.markdown("### Jackknife Bias Correction")

                        if statistic_type == "Variance estimate":
                            # Generate data with known variance
                            true_variance = 4.0
                            data = np.random.normal(0, np.sqrt(true_variance), size=jackknife_sample_size)

                            # Naive estimate (biased)
                            naive_var = np.var(data)  # Biased estimator (divide by n)

                            # Corrected estimate (unbiased)
                            corrected_var = np.var(data, ddof=1)  # Unbiased estimator (divide by n-1)

                            # Jackknife estimate
                            jackknife_vars = []
                            for i in range(jackknife_sample_size):
                                leave_one_out = np.delete(data, i)
                                jackknife_vars.append(np.var(leave_one_out))

                            jack_mean = np.mean(jackknife_vars)
                            jack_bias = (jackknife_sample_size - 1) * (jack_mean - naive_var)
                            jackknife_var = naive_var - jack_bias

                            # Plotting
                            fig, ax = plt.subplots(figsize=(10, 6))

                            ax.bar(['Naive (biased)', 'Unbiased', 'Jackknife'],
                                   [naive_var, corrected_var, jackknife_var],
                                   color=['salmon', 'lightgreen', 'skyblue'])
                            ax.axhline(true_variance, color='r', linestyle='--',
                                       label=f'True variance: {true_variance}')
                            ax.set_ylabel('Variance estimate')
                            ax.set_title(f'Variance Estimation Comparison (n={jackknife_sample_size})')
                            ax.legend()

                            for i, v in enumerate([naive_var, corrected_var, jackknife_var]):
                                ax.text(i, v + 0.1, f'{v:.4f}', ha='center')

                            st.pyplot(fig)

                            # Explanation
                            st.markdown(f"""
                                    **Variance Estimation Results:**
                                    - True variance: {true_variance}
                                    - Naive (biased) estimator: {naive_var:.4f} (Bias: {naive_var - true_variance:.4f})
                                    - Unbiased estimator (n-1): {corrected_var:.4f} (Bias: {corrected_var - true_variance:.4f})
                                    - Jackknife estimator: {jackknife_var:.4f} (Bias: {jackknife_var - true_variance:.4f})

                                    The jackknife estimator approximates the behavior of the unbiased estimator in this case.
                                    As sample size increases, all estimators converge to the true variance.
                                    """)

                        elif statistic_type == "Correlation estimate":
                            # Generate data with known correlation
                            true_corr = 0.7
                            cov_matrix = [[1, true_corr], [true_corr, 1]]
                            data = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix,
                                                                 size=jackknife_sample_size)

                            # Sample correlation (potentially biased in small samples)
                            sample_corr = np.corrcoef(data.T)[0, 1]

                            # Jackknife correlation
                            jackknife_corrs = []
                            for i in range(jackknife_sample_size):
                                leave_one_out = np.delete(data, i, axis=0)
                                jackknife_corrs.append(np.corrcoef(leave_one_out.T)[0, 1])

                            jack_mean = np.mean(jackknife_corrs)
                            jack_bias = (jackknife_sample_size - 1) * (jack_mean - sample_corr)
                            jackknife_corr = sample_corr - jack_bias

                            # Plotting
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                            # Scatter plot
                            ax1.scatter(data[:, 0], data[:, 1], alpha=0.7)
                            ax1.set_xlabel('X')
                            ax1.set_ylabel('Y')
                            ax1.set_title(f'Sample Data (n={jackknife_sample_size})')

                            # Estimates comparison
                            ax2.bar(['Sample', 'Jackknife'], [sample_corr, jackknife_corr],
                                    color=['salmon', 'skyblue'])
                            ax2.axhline(true_corr, color='r', linestyle='--', label=f'True correlation: {true_corr}')
                            ax2.set_ylabel('Correlation estimate')
                            ax2.set_title('Correlation Estimation Comparison')
                            ax2.legend()

                            for i, v in enumerate([sample_corr, jackknife_corr]):
                                ax2.text(i, v + 0.02, f'{v:.4f}', ha='center')

                            plt.tight_layout()
                            st.pyplot(fig)

                            # Explanation
                            st.markdown(f"""
                                    **Correlation Estimation Results:**
                                    - True correlation: {true_corr}
                                    - Sample correlation: {sample_corr:.4f} (Bias: {sample_corr - true_corr:.4f})
                                    - Jackknife correlation: {jackknife_corr:.4f} (Bias: {jackknife_corr - true_corr:.4f})

                                    The jackknife method can help reduce bias in correlation estimates, especially in small samples.
                                    However, its effectiveness varies depending on the underlying data distribution.
                                    """)

                        elif statistic_type == "R-squared estimate":
                            # Generate regression data
                            X = np.random.normal(0, 1, (jackknife_sample_size, 3))
                            beta = np.array([0.5, 1.0, -0.7])

                            # True model has these variables plus error
                            Y = X @ beta + np.random.normal(0, 1, jackknife_sample_size)

                            # Calculate population R^2 (approximation)
                            var_y_model = np.var(X @ beta)
                            var_error = 1.0  # Error variance
                            true_r2 = var_y_model / (var_y_model + var_error)

                            # Sample R^2 (potentially optimistic/biased in small samples)
                            X_design = sm.add_constant(X)
                            model = sm.OLS(Y, X_design)
                            results = model.fit()
                            sample_r2 = results.rsquared

                            # Adjusted R^2 (common correction)
                            adj_r2 = results.rsquared_adj

                            # Jackknife R^2
                            jackknife_r2s = []
                            for i in range(jackknife_sample_size):
                                leave_one_out_X = np.delete(X_design, i, axis=0)
                                leave_one_out_Y = np.delete(Y, i)

                                loo_model = sm.OLS(leave_one_out_Y, leave_one_out_X)
                                loo_results = loo_model.fit()
                                jackknife_r2s.append(loo_results.rsquared)

                            jack_mean = np.mean(jackknife_r2s)
                            jack_bias = (jackknife_sample_size - 1) * (jack_mean - sample_r2)
                            jackknife_r2 = sample_r2 - jack_bias

                            # Plotting
                            fig, ax = plt.subplots(figsize=(10, 6))

                            ax.bar(['Sample R²', 'Adjusted R²', 'Jackknife R²'],
                                   [sample_r2, adj_r2, jackknife_r2],
                                   color=['salmon', 'lightgreen', 'skyblue'])
                            ax.axhline(true_r2, color='r', linestyle='--', label=f'True R²: {true_r2:.4f}')
                            ax.set_ylabel('R-squared estimate')
                            ax.set_title(f'R-squared Estimation Comparison (n={jackknife_sample_size})')
                            ax.legend()

                            for i, v in enumerate([sample_r2, adj_r2, jackknife_r2]):
                                ax.text(i, v + 0.02, f'{v:.4f}', ha='center')

                            st.pyplot(fig)

                            # Explanation
                            st.markdown(f"""
                                    **R-squared Estimation Results:**
                                    - Approximate true R²: {true_r2:.4f}
                                    - Sample R²: {sample_r2:.4f} (Bias: {sample_r2 - true_r2:.4f})
                                    - Adjusted R²: {adj_r2:.4f} (Bias: {adj_r2 - true_r2:.4f})
                                    - Jackknife R²: {jackknife_r2:.4f} (Bias: {jackknife_r2 - true_r2:.4f})

                                    R-squared tends to be optimistically biased in small samples. Both adjusted R² and
                                    jackknife R² attempt to correct this bias, with different approaches.
                                    """)

                    else:  # K-fold cross-validation
                        st.markdown("### K-Fold Cross-Validation")

                        # Generate regression data
                        np.random.seed(42)  # For reproducibility
                        X = np.random.normal(0, 1, (jackknife_sample_size, 4))
                        beta = np.array([1.0, 0.8, 0.0, 0.0])  # First two variables are relevant
                        Y = X @ beta + np.random.normal(0, 2, jackknife_sample_size)

                        # Prepare models
                        if model_type == "Linear regression":
                            model = LinearRegression()
                            model_name = "OLS"
                        else:  # Ridge regression
                            model = Ridge(alpha=1.0)
                            model_name = "Ridge"

                        # K-fold cross-validation
                        from sklearn.model_selection import KFold

                        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

                        train_errors = []
                        test_errors = []
                        train_r2s = []
                        test_r2s = []
                        fold_data = []

                        # For plotting
                        all_preds = []
                        all_trues = []
                        all_fold_indices = []

                        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
                            X_train, X_test = X[train_idx], X[test_idx]
                            Y_train, Y_test = Y[train_idx], Y[test_idx]

                            model.fit(X_train, Y_train)

                            # Training metrics
                            train_pred = model.predict(X_train)
                            train_mse = np.mean((Y_train - train_pred) ** 2)
                            train_r2 = 1 - (
                                        np.sum((Y_train - train_pred) ** 2) / np.sum((Y_train - np.mean(Y_train)) ** 2))

                            # Test metrics
                            test_pred = model.predict(X_test)
                            test_mse = np.mean((Y_test - test_pred) ** 2)
                            test_r2 = 1 - (np.sum((Y_test - test_pred) ** 2) / np.sum((Y_test - np.mean(Y_test)) ** 2))

                            train_errors.append(train_mse)
                            test_errors.append(test_mse)
                            train_r2s.append(train_r2)
                            test_r2s.append(test_r2)

                            fold_data.append({
                                'Fold': i + 1,
                                'Train MSE': train_mse,
                                'Test MSE': test_mse,
                                'Train R²': train_r2,
                                'Test R²': test_r2,
                                'Samples': len(train_idx)
                            })

                            # Store predictions and true values
                            all_preds.extend(test_pred)
                            all_trues.extend(Y_test)
                            all_fold_indices.extend([i + 1] * len(test_idx))

                        # Create a DataFrame for the fold data
                        fold_df = pd.DataFrame(fold_data)

                        # Calculate averages
                        avg_train_mse = np.mean(train_errors)
                        avg_test_mse = np.mean(test_errors)
                        avg_train_r2 = np.mean(train_r2s)
                        avg_test_r2 = np.mean(test_r2s)

                        # Fit model on all data for comparison
                        full_model = model.fit(X, Y)
                        full_pred = full_model.predict(X)
                        full_mse = np.mean((Y - full_pred) ** 2)
                        full_r2 = 1 - (np.sum((Y - full_pred) ** 2) / np.sum((Y - np.mean(Y)) ** 2))

                        # Plotting
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                        # MSE comparison by fold
                        ax1.plot(fold_df['Fold'], fold_df['Train MSE'], 'o-', label='Training MSE')
                        ax1.plot(fold_df['Fold'], fold_df['Test MSE'], 's-', label='Test MSE')
                        ax1.axhline(full_mse, color='r', linestyle='--', label='Full data MSE')
                        ax1.set_xlabel('Fold')
                        ax1.set_ylabel('Mean Squared Error')
                        ax1.set_title(f'{k_folds}-Fold Cross-Validation: MSE')
                        ax1.legend()

                        # R² comparison by fold
                        ax2.plot(fold_df['Fold'], fold_df['Train R²'], 'o-', label='Training R²')
                        ax2.plot(fold_df['Fold'], fold_df['Test R²'], 's-', label='Test R²')
                        ax2.axhline(full_r2, color='r', linestyle='--', label='Full data R²')
                        ax2.set_xlabel('Fold')
                        ax2.set_ylabel('R-squared')
                        ax2.set_title(f'{k_folds}-Fold Cross-Validation: R²')
                        ax2.legend()

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Create a prediction vs true plot
                        fig2, ax = plt.subplots(figsize=(10, 6))

                        for fold in range(1, k_folds + 1):
                            mask = np.array(all_fold_indices) == fold
                            ax.scatter(np.array(all_trues)[mask], np.array(all_preds)[mask],
                                       alpha=0.7, label=f'Fold {fold}')

                        ax.plot([min(all_trues), max(all_trues)], [min(all_trues), max(all_trues)], 'k--', alpha=0.5)
                        ax.set_xlabel('True Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title(f'{model_name} Regression: Predicted vs True Values')
                        ax.legend()

                        st.pyplot(fig2)

                        # Results summary
                        st.markdown(f"""
                                **Cross-Validation Results Summary:**

                                | Metric | Training | Test | Full Data |
                                |--------|----------|------|-----------|
                                | Average MSE | {avg_train_mse:.4f} | {avg_test_mse:.4f} | {full_mse:.4f} |
                                | Average R² | {avg_train_r2:.4f} | {avg_test_r2:.4f} | {full_r2:.4f} |

                                **Key insights:**
                                1. The gap between training and test performance indicates the degree of overfitting
                                2. Higher variability across folds suggests model instability in small samples
                                3. Cross-validation provides a more realistic estimate of model performance than 
                                   single-sample metrics
                                4. {model_name} regression {'with regularization ' if model_type == 'Ridge regression' else ''}
                                   {'helps mitigate overfitting in small samples' if model_type == 'Ridge regression' else 'shows evidence of overfitting in this small sample'}
                                """)

        # PAGE: Bayesian Approaches
        ###########################################
elif selected_page == "Bayesian Approaches":
        st.title("Bayesian Approaches to Small Sample Problems")

        st.markdown("""
                                Bayesian methods provide a principled way to incorporate prior information and 
                                handle uncertainty in small samples. This section explores how Bayesian approaches
                                compare to frequentist methods when data is limited.
                                """)

        tab1, tab2 = st.tabs(["Bayesian vs. Frequentist Estimation", "Prior Sensitivity"])

        with tab1:
            st.subheader("Bayesian vs. Frequentist Estimation")

            st.markdown("""
                                    This tool compares Bayesian and frequentist estimation approaches in small samples,
                                    demonstrating how incorporating prior information can improve estimation.
                                    """)

            col1, col2 = st.columns([1, 2])

            with col1:
                bayesian_parameter = st.radio(
                    "Parameter to estimate:",
                    ["Mean", "Proportion", "Regression coefficient"]
                )

                if bayesian_parameter == "Mean":
                    true_mean = st.slider(
                        "True population mean",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        step=5.0
                    )

                    true_std = st.slider(
                        "True population std. dev.",
                        min_value=1.0,
                        max_value=30.0,
                        value=15.0,
                        step=1.0
                    )

                    prior_mean = st.slider(
                        "Prior mean",
                        min_value=0.0,
                        max_value=100.0,
                        value=60.0,
                        step=5.0
                    )

                    prior_std = st.slider(
                        "Prior std. dev.",
                        min_value=1.0,
                        max_value=50.0,
                        value=20.0,
                        step=5.0
                    )

                elif bayesian_parameter == "Proportion":
                    true_prop = st.slider(
                        "True population proportion",
                        min_value=0.05,
                        max_value=0.95,
                        value=0.3,
                        step=0.05
                    )

                    alpha_prior = st.slider(
                        "Beta prior: α parameter",
                        min_value=0.5,
                        max_value=10.0,
                        value=2.0,
                        step=0.5
                    )

                    beta_prior = st.slider(
                        "Beta prior: β parameter",
                        min_value=0.5,
                        max_value=10.0,
                        value=5.0,
                        step=0.5
                    )

                elif bayesian_parameter == "Regression coefficient":
                    true_coef = st.slider(
                        "True coefficient value",
                        min_value=-5.0,
                        max_value=5.0,
                        value=2.0,
                        step=0.5
                    )

                    prior_coef_mean = st.slider(
                        "Prior coefficient mean",
                        min_value=-5.0,
                        max_value=5.0,
                        value=1.0,
                        step=0.5
                    )

                    prior_coef_std = st.slider(
                        "Prior coefficient std. dev.",
                        min_value=0.5,
                        max_value=10.0,
                        value=2.0,
                        step=0.5
                    )

                    noise_level = st.slider(
                        "Noise level (std. dev.)",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.5,
                        step=0.5
                    )

                bayesian_sample_size = st.slider(
                    "Sample size",
                    min_value=5,
                    max_value=100,
                    value=15,
                    step=5
                )

                compare_approaches = st.button("Compare Approaches")

            with col2:
                if compare_approaches:
                    if bayesian_parameter == "Mean":
                        st.markdown("### Estimating Population Mean")

                        # Generate data from true distribution
                        data = np.random.normal(true_mean, true_std, size=bayesian_sample_size)

                        # Frequentist estimate
                        freq_mean = np.mean(data)
                        freq_std = np.std(data, ddof=1)
                        freq_se = freq_std / np.sqrt(bayesian_sample_size)
                        freq_ci = (freq_mean - 1.96 * freq_se, freq_mean + 1.96 * freq_se)

                        # Bayesian posterior
                        # For normal with known variance, posterior mean is weighted average
                        likelihood_precision = bayesian_sample_size / (true_std ** 2)
                        prior_precision = 1 / (prior_std ** 2)
                        posterior_precision = likelihood_precision + prior_precision

                        posterior_mean = (
                                                     likelihood_precision * freq_mean + prior_precision * prior_mean) / posterior_precision
                        posterior_std = np.sqrt(1 / posterior_precision)
                        posterior_ci = (posterior_mean - 1.96 * posterior_std, posterior_mean + 1.96 * posterior_std)

                        # Plotting
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Data points
                        ax.scatter(np.arange(bayesian_sample_size), data, color='black', alpha=0.7, label='Data points')

                        # True mean
                        ax.axhline(true_mean, color='green', linestyle='-', linewidth=2,
                                   label=f'True mean: {true_mean}')

                        # Frequentist estimate & CI
                        ax.axhline(freq_mean, color='blue', linestyle='--', label=f'Freq mean: {freq_mean:.2f}')
                        ax.fill_between([-1, bayesian_sample_size],
                                        [freq_ci[0], freq_ci[0]],
                                        [freq_ci[1], freq_ci[1]],
                                        color='blue', alpha=0.2,
                                        label=f'Freq 95% CI: [{freq_ci[0]:.2f}, {freq_ci[1]:.2f}]')

                        # Bayesian estimate & CI
                        ax.axhline(posterior_mean, color='red', linestyle='--',
                                   label=f'Bayes mean: {posterior_mean:.2f}')
                        ax.fill_between([-1, bayesian_sample_size],
                                        [posterior_ci[0], posterior_ci[0]],
                                        [posterior_ci[1], posterior_ci[1]],
                                        color='red', alpha=0.2,
                                        label=f'Bayes 95% CI: [{posterior_ci[0]:.2f}, {posterior_ci[1]:.2f}]')

                        ax.set_xlabel('Observation index')
                        ax.set_ylabel('Value')
                        ax.set_title(f'Normal Mean Estimation (n={bayesian_sample_size})')
                        ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=2)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Prior, likelihood, and posterior distributions
                        x = np.linspace(
                            min(prior_mean - 3 * prior_std, freq_mean - 3 * freq_std),
                            max(prior_mean + 3 * prior_std, freq_mean + 3 * freq_std),
                            1000
                        )

                        prior_pdf = stats.norm.pdf(x, prior_mean, prior_std)
                        likelihood_pdf = stats.norm.pdf(x, freq_mean, freq_std / np.sqrt(bayesian_sample_size))
                        posterior_pdf = stats.norm.pdf(x, posterior_mean, posterior_std)

                        fig2, ax2 = plt.subplots(figsize=(10, 6))

                        ax2.plot(x, prior_pdf, 'b--', label=f'Prior: N({prior_mean}, {prior_std}²)')
                        ax2.plot(x, likelihood_pdf, 'g-', label=f'Likelihood: N({freq_mean:.2f}, {freq_se:.2f}²)')
                        ax2.plot(x, posterior_pdf, 'r-', linewidth=2,
                                 label=f'Posterior: N({posterior_mean:.2f}, {posterior_std:.2f}²)')
                        ax2.axvline(true_mean, color='black', linestyle='-', label=f'True mean: {true_mean}')

                        ax2.set_xlabel('Mean value')
                        ax2.set_ylabel('Density')
                        ax2.set_title('Prior, Likelihood, and Posterior Distributions')
                        ax2.legend()

                        st.pyplot(fig2)

                        # Metrics comparison
                        freq_mse = (freq_mean - true_mean) ** 2
                        bayes_mse = (posterior_mean - true_mean) ** 2

                        st.markdown(f"""
                                    **Comparison of Estimation Approaches:**

                                    | Metric | Frequentist | Bayesian |
                                    |--------|-------------|----------|
                                    | Estimate | {freq_mean:.4f} | {posterior_mean:.4f} |
                                    | Standard Error | {freq_se:.4f} | {posterior_std:.4f} |
                                    | 95% CI Width | {freq_ci[1] - freq_ci[0]:.4f} | {posterior_ci[1] - posterior_ci[0]:.4f} |
                                    | Squared Error | {freq_mse:.4f} | {bayes_mse:.4f} |
                                    | True Value in CI | {'✅' if freq_ci[0] <= true_mean <= freq_ci[1] else '❌'} | {'✅' if posterior_ci[0] <= true_mean <= posterior_ci[1] else '❌'} |

                                    **Analysis:**
                                    - The Bayesian approach incorporates prior information, resulting in a posterior that balances the prior and the data.
                                    - As sample size increases, the likelihood dominates and both approaches converge.
                                    - In this small sample, the Bayesian CI is narrower due to the additional information from the prior.
                                    - The {'Bayesian' if bayes_mse < freq_mse else 'Frequentist'} estimate has lower squared error in this particular sample.
                                    """)

                    elif bayesian_parameter == "Proportion":
                        st.markdown("### Estimating Population Proportion")

                        # Generate binary data
                        data = np.random.binomial(1, true_prop, size=bayesian_sample_size)
                        successes = np.sum(data)

                        # Frequentist estimate
                        freq_prop = successes / bayesian_sample_size
                        freq_se = np.sqrt(freq_prop * (1 - freq_prop) / bayesian_sample_size)
                        freq_ci = (max(0, freq_prop - 1.96 * freq_se), min(1, freq_prop + 1.96 * freq_se))

                        # Bayesian estimate (Beta-Binomial)
                        posterior_alpha = alpha_prior + successes
                        posterior_beta = beta_prior + bayesian_sample_size - successes

                        bayes_prop = posterior_alpha / (posterior_alpha + posterior_beta)  # Posterior mean
                        bayes_ci = stats.beta.ppf([0.025, 0.975], posterior_alpha, posterior_beta)

                        # Prior mean and CI
                        prior_prop = alpha_prior / (alpha_prior + beta_prior)
                        prior_ci = stats.beta.ppf([0.025, 0.975], alpha_prior, beta_prior)

                        # Plotting
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                        # Bar chart of data
                        ax1.bar(['Success', 'Failure'], [successes, bayesian_sample_size - successes],
                                color=['green', 'red'], alpha=0.7)
                        ax1.set_ylabel('Count')
                        ax1.set_title(f'Sample Data: {successes} successes in {bayesian_sample_size} trials')

                        for i, count in enumerate([successes, bayesian_sample_size - successes]):
                            ax1.text(i, count + 0.5, str(count), ha='center')

                        # Prior, posterior distributions
                        x = np.linspace(0, 1, 1000)
                        prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
                        posterior_pdf = stats.beta.pdf(x, posterior_alpha, posterior_beta)

                        ax2.plot(x, prior_pdf, 'b--', label=f'Prior: Beta({alpha_prior}, {beta_prior})')
                        ax2.plot(x, posterior_pdf, 'r-', linewidth=2,
                                 label=f'Posterior: Beta({posterior_alpha:.1f}, {posterior_beta:.1f})')
                        ax2.axvline(true_prop, color='black', linestyle='-', label=f'True proportion: {true_prop}')
                        ax2.axvline(freq_prop, color='green', linestyle='--', label=f'Freq. estimate: {freq_prop:.3f}')

                        ax2.fill_between(x, 0, posterior_pdf, where=(x >= bayes_ci[0]) & (x <= bayes_ci[1]),
                                         color='red', alpha=0.2,
                                         label=f'Bayes 95% CI: [{bayes_ci[0]:.3f}, {bayes_ci[1]:.3f}]')

                        ax2.set_xlabel('Proportion')
                        ax2.set_ylabel('Density')
                        ax2.set_title('Prior and Posterior Distributions')
                        ax2.legend(loc='best')

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Metrics comparison
                        freq_mse = (freq_prop - true_prop) ** 2
                        bayes_mse = (bayes_prop - true_prop) ** 2

                        st.markdown(f"""
                                    **Comparison of Proportion Estimation Approaches:**

                                    | Metric | Frequentist | Bayesian |
                                    |--------|-------------|----------|
                                    | Estimate | {freq_prop:.4f} | {bayes_prop:.4f} |
                                    | Prior Estimate | N/A | {prior_prop:.4f} |
                                    | Standard Error | {freq_se:.4f} | {np.sqrt(bayes_prop * (1 - bayes_prop) / (posterior_alpha + posterior_beta + 1)):.4f} |
                                    | 95% CI | [{freq_ci[0]:.4f}, {freq_ci[1]:.4f}] | [{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}] |
                                    | Squared Error | {freq_mse:.4f} | {bayes_mse:.4f} |
                                    | True Value in CI | {'✅' if freq_ci[0] <= true_prop <= freq_ci[1] else '❌'} | {'✅' if bayes_ci[0] <= true_prop <= bayes_ci[1] else '❌'} |

                                    **Analysis:**
                                    - The Beta prior combined with Binomial likelihood gives a Beta posterior distribution.
                                    - The Bayesian approach shrinks the estimate toward the prior mean of {prior_prop:.4f}.
                                    - For proportions near 0 or 1, Bayesian approaches are particularly valuable as they avoid boundary issues.
                                    - The {'Bayesian' if bayes_mse < freq_mse else 'Frequentist'} estimate has lower squared error in this sample.
                                    """)

                    elif bayesian_parameter == "Regression coefficient":
                        st.markdown("### Estimating Regression Coefficient")

                        # Generate regression data
                        X = np.random.normal(0, 1, bayesian_sample_size)
                        Y = true_coef * X + np.random.normal(0, noise_level, bayesian_sample_size)

                        # Frequentist OLS
                        X_design = sm.add_constant(X)
                        ols_model = sm.OLS(Y, X_design)
                        ols_results = ols_model.fit()

                        freq_coef = ols_results.params[1]
                        freq_se = ols_results.bse[1]
                        freq_ci = ols_results.conf_int(alpha=0.05)[1]

                        # Bayesian regression (analytical for known variance case)
                        # For simplicity, we'll implement a conjugate normal prior on the coefficient

                        # Precision matrices (inverse covariance)
                        prior_precision = 1 / prior_coef_std ** 2

                        # Design matrix X'X
                        XtX = X_design.T @ X_design

                        # Prior mean vector (for intercept and slope)
                        prior_mean_vec = np.array([0, prior_coef_mean])

                        # Posterior precision = prior precision + data precision
                        posterior_precision = XtX / noise_level ** 2 + np.diag([prior_precision / 100, prior_precision])

                        # Posterior mean = posterior precision^(-1) * (prior precision * prior mean + X'Y/sigma^2)
                        XtY = X_design.T @ Y
                        posterior_mean_vec = np.linalg.solve(
                            posterior_precision,
                            np.diag([prior_precision / 100, prior_precision]) @ prior_mean_vec + XtY / noise_level ** 2
                        )

                        bayes_coef = posterior_mean_vec[1]
                        posterior_var = np.linalg.inv(posterior_precision)[1, 1]
                        bayes_se = np.sqrt(posterior_var)
                        bayes_ci = (bayes_coef - 1.96 * bayes_se, bayes_coef + 1.96 * bayes_se)

                        # Plotting
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                        # Scatter plot with regression lines
                        ax1.scatter(X, Y, alpha=0.7, label='Data')

                        # True regression line
                        x_range = np.linspace(min(X), max(X), 100)
                        ax1.plot(x_range, true_coef * x_range, 'g-', label=f'True: β = {true_coef}')

                        # Frequentist regression line
                        ax1.plot(x_range, ols_results.params[0] + freq_coef * x_range, 'b--',
                                 label=f'OLS: β = {freq_coef:.3f}')

                        # Bayesian regression line
                        ax1.plot(x_range, posterior_mean_vec[0] + bayes_coef * x_range, 'r--',
                                 label=f'Bayes: β = {bayes_coef:.3f}')

                        ax1.set_xlabel('X')
                        ax1.set_ylabel('Y')
                        ax1.set_title(f'Regression Data (n={bayesian_sample_size})')
                        ax1.legend()

                        # Coefficient distributions
                        coef_range = np.linspace(
                            min(prior_coef_mean - 3 * prior_coef_std, freq_coef - 3 * freq_se),
                            max(prior_coef_mean + 3 * prior_coef_std, freq_coef + 3 * freq_se),
                            1000
                        )

                        prior_pdf = stats.norm.pdf(coef_range, prior_coef_mean, prior_coef_std)
                        likelihood_pdf = stats.norm.pdf(coef_range, freq_coef, freq_se)
                        posterior_pdf = stats.norm.pdf(coef_range, bayes_coef, bayes_se)

                        ax2.plot(coef_range, prior_pdf, 'b--', label=f'Prior: N({prior_coef_mean}, {prior_coef_std}²)')
                        ax2.plot(coef_range, likelihood_pdf, 'g-',
                                 label=f'Likelihood: N({freq_coef:.3f}, {freq_se:.3f}²)')
                        ax2.plot(coef_range, posterior_pdf, 'r-', linewidth=2,
                                 label=f'Posterior: N({bayes_coef:.3f}, {bayes_se:.3f}²)')
                        ax2.axvline(true_coef, color='black', linestyle='-', label=f'True β: {true_coef}')

                        ax2.set_xlabel('Coefficient value')
                        ax2.set_ylabel('Density')
                        ax2.set_title('Prior, Likelihood, and Posterior Distributions')
                        ax2.legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=2)

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Metrics comparison
                        freq_mse = (freq_coef - true_coef) ** 2
                        bayes_mse = (bayes_coef - true_coef) ** 2

                        st.markdown(f"""
                                    **Comparison of Regression Coefficient Estimation:**

                                    | Metric | Frequentist (OLS) | Bayesian |
                                    |--------|-------------|----------|
                                    | Coefficient | {freq_coef:.4f} | {bayes_coef:.4f} |
                                    | Standard Error | {freq_se:.4f} | {bayes_se:.4f} |
                                    | 95% CI | [{freq_ci[0]:.4f}, {freq_ci[1]:.4f}] | [{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}] |
                                    | CI Width | {freq_ci[1] - freq_ci[0]:.4f} | {bayes_ci[1] - bayes_ci[0]:.4f} |
                                    | Squared Error | {freq_mse:.4f} | {bayes_mse:.4f} |
                                    | True Value in CI | {'✅' if freq_ci[0] <= true_coef <= freq_ci[1] else '❌'} | {'✅' if bayes_ci[0] <= true_coef <= bayes_ci[1] else '❌'} |

                                    **Analysis:**
                                    - The Bayesian approach pulls the estimate toward the prior mean of {prior_coef_mean}.
                                    - With a prior centered near the true value, Bayesian estimation can greatly improve accuracy.
                                    - In small samples, the prior can dominate; as sample size increases, the likelihood dominates.
                                    - The {'Bayesian' if bayes_mse < freq_mse else 'Frequentist'} estimate has lower squared error in this sample.

                                    **Note:** This simplified example uses a conjugate normal prior with known error variance.
                                    Full Bayesian regression typically uses MCMC sampling to handle unknown variances and more complex priors.
                                    """)

        with tab2:
            st.subheader("Prior Sensitivity Analysis")

            st.markdown("""
                                    The choice of prior can significantly impact Bayesian inference in small samples.
                                    This tool demonstrates the sensitivity of posterior estimates to different prior specifications.
                                    """)

            col1, col2 = st.columns([1, 2])

            with col1:
                prior_sensitivity_param = st.radio(
                    "Parameter to analyze:",
                    ["Mean", "Proportion"]
                )

                if prior_sensitivity_param == "Mean":
                    true_mean_sens = st.slider(
                        "True population mean",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        step=5.0,
                        key="sens_true_mean"
                    )

                    true_std_sens = st.slider(
                        "True population std. dev.",
                        min_value=1.0,
                        max_value=30.0,
                        value=15.0,
                        step=1.0,
                        key="sens_true_std"
                    )

                    # Multiple prior means to compare
                    prior_mean_1 = st.slider(
                        "Prior 1: Mean",
                        min_value=0.0,
                        max_value=100.0,
                        value=30.0,
                        step=5.0
                    )

                    prior_mean_2 = st.slider(
                        "Prior 2: Mean",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        step=5.0
                    )

                    prior_mean_3 = st.slider(
                        "Prior 3: Mean",
                        min_value=0.0,
                        max_value=100.0,
                        value=70.0,
                        step=5.0
                    )

                    # Prior precision (inverse variance) instead of std dev for more intuitive scaling
                    prior_precision = st.slider(
                        "Prior precision (higher = stronger prior)",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.01,
                        step=0.01
                    )

                elif prior_sensitivity_param == "Proportion":
                    true_prop_sens = st.slider(
                        "True population proportion",
                        min_value=0.05,
                        max_value=0.95,
                        value=0.3,
                        step=0.05,
                        key="sens_true_prop"
                    )

                    # Beta prior parameters for multiple priors
                    # Prior 1: Weak/uninformative
                    alpha_1 = st.slider(
                        "Prior 1: α parameter (uninformative)",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.0,
                        step=0.5
                    )

                    beta_1 = st.slider(
                        "Prior 1: β parameter (uninformative)",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.0,
                        step=0.5
                    )

                    # Prior 2: Informative and accurate
                    alpha_2 = st.slider(
                        "Prior 2: α parameter (accurate)",
                        min_value=1.0,
                        max_value=20.0,
                        value=6.0,
                        step=1.0
                    )

                    beta_2 = st.slider(
                        "Prior 2: β parameter (accurate)",
                        min_value=1.0,
                        max_value=20.0,
                        value=14.0,
                        step=1.0
                    )

                    # Prior 3: Informative but misleading
                    alpha_3 = st.slider(
                        "Prior 3: α parameter (misleading)",
                        min_value=1.0,
                        max_value=20.0,
                        value=14.0,
                        step=1.0
                    )

                    beta_3 = st.slider(
                        "Prior 3: β parameter (misleading)",
                        min_value=1.0,
                        max_value=20.0,
                        value=6.0,
                        step=1.0
                    )

                prior_sens_sample_sizes = st.multiselect(
                    "Sample sizes to compare",
                    options=[5, 10, 20, 50, 100],
                    default=[5, 20, 50]
                )

                run_sensitivity = st.button("Run Sensitivity Analysis")

            with col2:
                if run_sensitivity:
                    if prior_sensitivity_param == "Mean":
                        st.markdown("### Prior Sensitivity Analysis for Mean Estimation")

                        # Convert precision to std dev for calculations
                        prior_std_1 = 1.0 / np.sqrt(prior_precision)
                        prior_std_2 = prior_std_1
                        prior_std_3 = prior_std_1

                        # Create a figure with subplots for each sample size
                        fig, axes = plt.subplots(len(prior_sens_sample_sizes), 1,
                                                 figsize=(10, 4 * len(prior_sens_sample_sizes)))

                        if len(prior_sens_sample_sizes) == 1:
                            axes = [axes]

                        # For tracking results across sample sizes
                        results_data = []

                        for i, sample_size in enumerate(sorted(prior_sens_sample_sizes)):
                            # Generate data
                            np.random.seed(42 + sample_size)  # Different but reproducible data for each size
                            data = np.random.normal(true_mean_sens, true_std_sens, size=sample_size)

                            # Sample statistics
                            sample_mean = np.mean(data)
                            sample_std = np.std(data, ddof=1)
                            sample_se = sample_std / np.sqrt(sample_size)

                            # Frequentist results
                            freq_ci = (sample_mean - 1.96 * sample_se, sample_mean + 1.96 * sample_se)

                            # Bayesian posteriors for each prior
                            posteriors = []
                            for prior_mean, prior_std in [(prior_mean_1, prior_std_1),
                                                          (prior_mean_2, prior_std_2),
                                                          (prior_mean_3, prior_std_3)]:
                                # Calculate posterior parameters
                                likelihood_precision = sample_size / (true_std_sens ** 2)
                                prior_precision = 1 / (prior_std ** 2)
                                posterior_precision = likelihood_precision + prior_precision

                                posterior_mean = (
                                                             likelihood_precision * sample_mean + prior_precision * prior_mean) / posterior_precision
                                posterior_std = np.sqrt(1 / posterior_precision)
                                posterior_ci = (
                                posterior_mean - 1.96 * posterior_std, posterior_mean + 1.96 * posterior_std)

                                posteriors.append({
                                    'mean': posterior_mean,
                                    'std': posterior_std,
                                    'ci': posterior_ci,
                                    'prior_mean': prior_mean,
                                    'prior_std': prior_std,
                                    'mse': (posterior_mean - true_mean_sens) ** 2
                                })

                                # Store results for table
                                results_data.append({
                                    'Sample Size': sample_size,
                                    'Prior Mean': prior_mean,
                                    'Posterior Mean': posterior_mean,
                                    'Posterior SD': posterior_std,
                                    'CI Width': posterior_ci[1] - posterior_ci[0],
                                    'MSE': (posterior_mean - true_mean_sens) ** 2,
                                    'Contains True': (posterior_ci[0] <= true_mean_sens <= posterior_ci[1])
                                })

                            # Plotting distributions
                            x = np.linspace(
                                min(prior_mean_1 - 3 * prior_std_1, sample_mean - 3 * sample_se),
                                max(prior_mean_3 + 3 * prior_std_3, sample_mean + 3 * sample_se),
                                1000
                            )

                            # Plot priors
                            axes[i].plot(x, stats.norm.pdf(x, prior_mean_1, prior_std_1), 'b--', alpha=0.5,
                                         label=f'Prior 1: N({prior_mean_1}, {prior_std_1:.1f}²)')
                            axes[i].plot(x, stats.norm.pdf(x, prior_mean_2, prior_std_2), 'g--', alpha=0.5,
                                         label=f'Prior 2: N({prior_mean_2}, {prior_std_2:.1f}²)')
                            axes[i].plot(x, stats.norm.pdf(x, prior_mean_3, prior_std_3), 'r--', alpha=0.5,
                                         label=f'Prior 3: N({prior_mean_3}, {prior_std_3:.1f}²)')

                            # Plot likelihood
                            axes[i].plot(x, stats.norm.pdf(x, sample_mean, sample_se), 'k-',
                                         label=f'Likelihood: N({sample_mean:.1f}, {sample_se:.1f}²)')

                            # Plot posteriors
                            axes[i].plot(x, stats.norm.pdf(x, posteriors[0]['mean'], posteriors[0]['std']), 'b-',
                                         label=f'Posterior 1: N({posteriors[0]["mean"]:.1f}, {posteriors[0]["std"]:.1f}²)')
                            axes[i].plot(x, stats.norm.pdf(x, posteriors[1]['mean'], posteriors[1]['std']), 'g-',
                                         label=f'Posterior 2: N({posteriors[1]["mean"]:.1f}, {posteriors[1]["std"]:.1f}²)')
                            axes[i].plot(x, stats.norm.pdf(x, posteriors[2]['mean'], posteriors[2]['std']), 'r-',
                                         label=f'Posterior 3: N({posteriors[2]["mean"]:.1f}, {posteriors[2]["std"]:.1f}²)')

                            # True mean line
                            axes[i].axvline(true_mean_sens, color='purple', linestyle='-', linewidth=2,
                                            label=f'True mean: {true_mean_sens}')

                            axes[i].set_title(f'Sample Size n={sample_size}')
                            axes[i].set_xlabel('Mean value')
                            axes[i].set_ylabel('Density')
                            axes[i].legend(loc='upper right')

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Create a DataFrame for the results
                        results_df = pd.DataFrame(results_data)

                        # Pivot the table for better display
                        pivot_df = results_df.pivot(index='Sample Size', columns='Prior Mean',
                                                    values=['Posterior Mean', 'Posterior SD', 'MSE', 'Contains True'])

                        st.markdown("### Results Summary")
                        st.dataframe(pivot_df)

                        # Convergence plot
                        fig2, ax = plt.subplots(figsize=(10, 6))

                        # Extract data for each prior
                        for prior_mean in [prior_mean_1, prior_mean_2, prior_mean_3]:
                            prior_data = results_df[results_df['Prior Mean'] == prior_mean]
                            ax.plot(prior_data['Sample Size'], prior_data['Posterior Mean'], 'o-',
                                    label=f'Prior mean = {prior_mean}')

                        ax.axhline(true_mean_sens, color='purple', linestyle='--',
                                   label=f'True mean = {true_mean_sens}')

                        ax.set_xlabel('Sample Size')
                        ax.set_ylabel('Posterior Mean')
                        ax.set_title('Convergence of Posterior Means with Increasing Sample Size')
                        ax.legend()

                        st.pyplot(fig2)

                        st.markdown(f"""
                                    **Key Insights from Prior Sensitivity Analysis:**

                                    1. **Prior Impact**: With small samples (n=5), the choice of prior has a substantial effect on posterior inference.

                                    2. **Convergence**: As sample size increases, all posteriors converge toward the true value regardless of prior.

                                    3. **Precision Trade-off**: More informative priors (higher precision) lead to narrower posterior intervals but 
                                       can increase bias if the prior is misspecified.

                                    4. **Sample Size Threshold**: In this example, by n=50 the likelihood largely dominates the priors, 
                                       making the choice of prior less critical.

                                    5. **Optimal Prior**: Prior 2 (mean={prior_mean_2}) performs best overall because it's closest to the true value.
                                    """)

                    elif prior_sensitivity_param == "Proportion":
                        st.markdown("### Prior Sensitivity Analysis for Proportion Estimation")

                        # Get prior means for labeling
                        prior_1_mean = alpha_1 / (alpha_1 + beta_1)
                        prior_2_mean = alpha_2 / (alpha_2 + beta_2)
                        prior_3_mean = alpha_3 / (alpha_3 + beta_3)

                        # Create a figure with subplots for each sample size
                        fig, axes = plt.subplots(len(prior_sens_sample_sizes), 1,
                                                 figsize=(10, 4 * len(prior_sens_sample_sizes)))

                        if len(prior_sens_sample_sizes) == 1:
                            axes = [axes]

                        # For tracking results across sample sizes
                        results_data = []

                        for i, sample_size in enumerate(sorted(prior_sens_sample_sizes)):
                            # Generate data
                            np.random.seed(42 + sample_size)  # Different but reproducible data for each size
                            data = np.random.binomial(1, true_prop_sens, size=sample_size)
                            successes = np.sum(data)

                            # Sample proportion
                            sample_prop = successes / sample_size
                            sample_se = np.sqrt(sample_prop * (1 - sample_prop) / sample_size)
                            freq_ci = (max(0, sample_prop - 1.96 * sample_se), min(1, sample_prop + 1.96 * sample_se))

                            # Bayesian posteriors for each prior
                            posteriors = []
                            for alpha, beta, prior_name, prior_mean in [
                                (alpha_1, beta_1, "Uninformative", prior_1_mean),
                                (alpha_2, beta_2, "Accurate", prior_2_mean),
                                (alpha_3, beta_3, "Misleading", prior_3_mean)
                            ]:
                                # Calculate posterior parameters
                                post_alpha = alpha + successes
                                post_beta = beta + sample_size - successes
                                post_mean = post_alpha / (post_alpha + post_beta)
                                post_ci = stats.beta.ppf([0.025, 0.975], post_alpha, post_beta)

                                posteriors.append({
                                    'mean': post_mean,
                                    'alpha': post_alpha,
                                    'beta': post_beta,
                                    'ci': post_ci,
                                    'prior_name': prior_name,
                                    'prior_mean': prior_mean,
                                    'mse': (post_mean - true_prop_sens) ** 2
                                })

                                # Store results for table
                                results_data.append({
                                    'Sample Size': sample_size,
                                    'Prior': prior_name,
                                    'Prior Mean': prior_mean,
                                    'Posterior Mean': post_mean,
                                    'CI Width': post_ci[1] - post_ci[0],
                                    'MSE': (post_mean - true_prop_sens) ** 2,
                                    'Contains True': (post_ci[0] <= true_prop_sens <= post_ci[1])
                                })

                            # Plotting distributions
                            x = np.linspace(0, 1, 1000)

                            # Plot priors
                            axes[i].plot(x, stats.beta.pdf(x, alpha_1, beta_1), 'b--', alpha=0.5,
                                         label=f'Prior 1 (Uninformative): Beta({alpha_1}, {beta_1})')
                            axes[i].plot(x, stats.beta.pdf(x, alpha_2, beta_2), 'g--', alpha=0.5,
                                         label=f'Prior 2 (Accurate): Beta({alpha_2}, {beta_2})')
                            axes[i].plot(x, stats.beta.pdf(x, alpha_3, beta_3), 'r--', alpha=0.5,
                                         label=f'Prior 3 (Misleading): Beta({alpha_3}, {beta_3})')

                            # Construct approximate likelihood using beta distribution
                            # This is just for visualization - the actual likelihood is binomial
                            if successes > 0 and successes < sample_size:
                                likelihood_alpha = successes + 0.5
                                likelihood_beta = sample_size - successes + 0.5
                                axes[i].plot(x, stats.beta.pdf(x, likelihood_alpha, likelihood_beta), 'k-',
                                             label=f'Approx. Likelihood: {successes}/{sample_size} = {sample_prop:.2f}')

                            # Plot posteriors
                            axes[i].plot(x, stats.beta.pdf(x, posteriors[0]['alpha'], posteriors[0]['beta']), 'b-',
                                         label=f'Posterior 1: Beta({posteriors[0]["alpha"]:.1f}, {posteriors[0]["beta"]:.1f})')
                            axes[i].plot(x, stats.beta.pdf(x, posteriors[1]['alpha'], posteriors[1]['beta']), 'g-',
                                         label=f'Posterior 2: Beta({posteriors[1]["alpha"]:.1f}, {posteriors[1]["beta"]:.1f})')
                            axes[i].plot(x, stats.beta.pdf(x, posteriors[2]['alpha'], posteriors[2]['beta']), 'r-',
                                         label=f'Posterior 3: Beta({posteriors[2]["alpha"]:.1f}, {posteriors[2]["beta"]:.1f})')

                            # True proportion line
                            axes[i].axvline(true_prop_sens, color='purple', linestyle='-', linewidth=2,
                                            label=f'True proportion: {true_prop_sens}')

                            axes[i].set_title(f'Sample Size n={sample_size}, Data: {successes} successes')
                            axes[i].set_xlabel('Proportion')
                            axes[i].set_ylabel('Density')
                            axes[i].legend(loc='best')

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Create a DataFrame for the results
                        results_df = pd.DataFrame(results_data)

                        # Pivot the table for better display
                        pivot_df = results_df.pivot(index='Sample Size', columns='Prior',
                                                    values=['Posterior Mean', 'MSE', 'CI Width', 'Contains True'])

                        st.markdown("### Results Summary")
                        st.dataframe(pivot_df)

                        # Convergence plot
                        fig2, ax = plt.subplots(figsize=(10, 6))

                        # Extract data for each prior
                        for prior in ["Uninformative", "Accurate", "Misleading"]:
                            prior_data = results_df[results_df['Prior'] == prior]
                            ax.plot(prior_data['Sample Size'], prior_data['Posterior Mean'], 'o-',
                                    label=f'Prior: {prior}')

                        ax.axhline(true_prop_sens, color='purple', linestyle='--',
                                   label=f'True proportion = {true_prop_sens}')

                        ax.set_xlabel('Sample Size')
                        ax.set_ylabel('Posterior Mean')
                        ax.set_title('Convergence of Posterior Means with Increasing Sample Size')
                        ax.legend()

                        st.pyplot(fig2)

                        st.markdown(f"""
                                    **Key Insights from Prior Sensitivity Analysis for Proportions:**

                                    1. **Prior Impact**: With small samples (n={min(prior_sens_sample_sizes)}), the choice of prior can drastically
                                       affect posterior inference for proportions.

                                    2. **Uninformative Prior**: The uniform prior (Beta(1,1)) puts equal weight on all possible proportion values,
                                       letting the data speak for itself. This is less helpful in very small samples.

                                    3. **Informative Priors**: 
                                       - The accurate prior (centered near {prior_2_mean:.2f}) improves estimation in small samples.
                                       - The misleading prior (centered near {prior_3_mean:.2f}) biases results away from the true value.

                                    4. **Convergence Rate**: By n={max(prior_sens_sample_sizes)}, posteriors have mostly converged to the true value,
                                       but the misleading prior still shows some bias.

                                    5. **Interval Width**: More informative priors produce narrower intervals, which can improve precision
                                       if the prior is well-specified, but may exclude the true value if the prior is wrong.
                                    """)

    # PAGE: Practical Guidelines
    ###########################################
elif selected_page == "Practical Guidelines":
    st.title("Practical Guidelines for Small Sample Econometrics")

    st.markdown("""
                                This section provides practical recommendations and best practices for econometric analysis 
                                when working with small samples. These guidelines synthesize the concepts explored throughout
                                this application.
                                """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Sample Size Planning")

        st.markdown("""
                                    **Power Analysis**

                                    Before collecting data, conduct power analysis to determine the minimum sample size 
                                    needed to detect effects of interest:

                                    * For regression: Calculate required n based on expected R², desired power (typically 0.8),
                                      significance level (typically 0.05), and number of predictors
                                    * For differences: Calculate required n based on expected effect size, desired power,
                                      and significance level

                                    **Data Collection Strategy**

                                    If facing sample size constraints:

                                    * Consider sequential testing approaches with interim analyses
                                    * Focus on more homogeneous populations to reduce variance
                                    * Collect more information per observation to improve precision
                                    * Consider alternative sampling strategies (stratified, cluster)

                                    **Expectations Management**

                                    Be realistic about what can be learned from small samples:

                                    * Set appropriate confidence level expectations (wider intervals)
                                    * Focus on larger effects that are detectable with available power
                                    * Consider exploratory rather than confirmatory frameworks
                                    """)

        st.subheader("Model Selection & Specification")

        st.markdown("""
                                    **Parsimony Principle**

                                    Keep models simple when samples are small:

                                    * Limit the number of predictors (rule of thumb: at least 10-15 observations per parameter)
                                    * Focus on variables with strong theoretical justification
                                    * Consider dimension reduction techniques (PCA, factor analysis)

                                    **Specification Testing**

                                    Be cautious with model selection in small samples:

                                    * Avoid excessive specification searches that capitalize on chance
                                    * Use information criteria (AIC, BIC) instead of nested hypothesis testing when possible
                                    * Consider the bias-variance tradeoff explicitly

                                    **Addressing Nonlinearity**

                                    When exploring nonlinear relationships:

                                    * Use theory to guide functional form rather than flexible specifications
                                    * Consider transformations (log, polynomial) over threshold or spline models
                                    * Visualize relationships before modeling to guide specification
                                    """)

    with col2:
        st.subheader("Estimation Approaches")

        st.markdown("""
                                    **Regularization Techniques**

                                    Use shrinkage/regularization methods to improve stability:

                                    * Ridge regression when multicollinearity is present
                                    * LASSO when variable selection is needed
                                    * Elastic net for a balanced approach
                                    * Calibrate regularization strength via cross-validation

                                    **Bayesian Methods**

                                    Consider Bayesian approaches to incorporate prior information:

                                    * Use informative priors based on previous studies or theory
                                    * Conduct sensitivity analysis with different prior specifications
                                    * Use hierarchical/multilevel models to partially pool information
                                    * Report posterior intervals rather than p-values

                                    **Resampling Methods**

                                    Use resampling to improve inference:

                                    * Bootstrap for more robust confidence intervals
                                    * Jackknife for bias correction
                                    * Cross-validation for model selection and validation
                                    * Permutation tests for hypothesis testing
                                    """)

        st.subheader("Inference & Reporting")

        st.markdown("""
                                    **Confidence Intervals & Uncertainty**

                                    Focus on precision and uncertainty:

                                    * Report confidence intervals rather than just point estimates
                                    * Consider reporting prediction intervals for forecasts
                                    * Use exact tests rather than asymptotic approximations when available
                                    * Report effect sizes alongside significance tests

                                    **Multiple Testing Concerns**

                                    Be vigilant about multiple testing issues:

                                    * Use family-wise error corrections (Bonferroni, Holm) or
                                      false discovery rate control (Benjamini-Hochberg)
                                    * Pre-register analyses when possible to avoid fishing expeditions
                                    * Distinguish between confirmatory and exploratory analyses

                                    **Transparent Reporting**

                                    Maintain transparency in research reports:

                                    * Acknowledge sample size limitations explicitly
                                    * Report all analyses conducted (not just "significant" results)
                                    * Conduct sensitivity analyses to test robustness of conclusions
                                    * Consider publishing data and code for reproducibility
                                    """)

    st.subheader("Decision Tree for Small Sample Econometric Analysis")

    decision_tree_code = """
                    digraph G {
                        node [shape=box, style="rounded,filled", fillcolor=lightblue, fontname="Arial"];
                        edge [fontname="Arial"];

                        // Decision nodes
                        start [label="Small Sample\nEconometric Analysis", fillcolor=lightgreen];
                        q1 [label="How small is\nyour sample?"];
                        q2a [label="Do you have prior\ninformation?"];
                        q2b [label="Are there many\npotential predictors?"];
                        q2c [label="Is normality\nassumption valid?"];

                        // Recommendation nodes
                        rec1a [label="Consider power analysis\nand data augmentation", fillcolor=lightyellow];
                        rec1b [label="Focus on key variables,\nuse regularization", fillcolor=lightyellow];
                        rec1c [label="Use nonparametric methods\nor robust estimators", fillcolor=lightyellow];

                        rec2a [label="Consider Bayesian methods\nwith informative priors", fillcolor=lightyellow];
                        rec2b [label="Use cross-validation and\nregularization (Ridge, LASSO)", fillcolor=lightyellow];
                        rec2c [label="Bootstrap for confidence\nintervals and inference", fillcolor=lightyellow];

                        rec3a [label="Report uncertainty,\nuse exact tests", fillcolor=lightyellow];
                        rec3b [label="Limit model complexity,\nuse information criteria", fillcolor=lightyellow];
                        rec3c [label="Examine residuals,\nuse robust standard errors", fillcolor=lightyellow];

                        // Edge connections
                        start -> q1;

                        q1 -> q2a [label="Extremely small\n(n < 30)"];
                        q1 -> q2b [label="Small\n(30 ≤ n < 100)"];
                        q1 -> q2c [label="Moderate\n(100 ≤ n < 200)"];

                        q2a -> rec1a [label="No"];
                        q2a -> rec2a [label="Yes"];
                        q2a -> rec3a [label="Limited"];

                        q2b -> rec1b [label="Yes"];
                        q2b -> rec2b [label="Some"];
                        q2b -> rec3b [label="No"];

                        q2c -> rec1c [label="No"];
                        q2c -> rec2c [label="Uncertain"];
                        q2c -> rec3c [label="Yes"];
                    }
                    """

    st.graphviz_chart(decision_tree_code)

    st.subheader("Sample Size Reference Table")

    sample_sizes = pd.DataFrame({
        "Analysis Type": [
            "Simple linear regression (1 predictor)",
            "Multiple regression (p predictors)",
            "Mediation analysis",
            "Moderator analysis (interaction)",
            "Logistic regression",
            "Time series forecasting (ARIMA)",
            "Panel data (fixed effects)",
            "Structural equation model (per latent variable)",
            "Propensity score matching",
            "Difference-in-differences"
        ],
        "Minimum Sample Size": [
            "15 observations",
            "10-15 observations per predictor",
            "50-100 observations",
            "Sample size 3-4× higher than main effect",
            "10-15 events per predictor",
            "50 observations (minimum), ideally 100+",
            "20 observations per group, 2+ time periods",
            "10-20 observations per parameter",
            "40 observations per group (minimum)",
            "50 observations per group, 2+ time periods"
        ],
        "Small Sample Adjustments": [
            "Use exact t-distribution, bootstrap CIs",
            "Regularization, variable selection, PCA",
            "Bootstrap confidence intervals, Bayesian estimation",
            "Specify interactions a priori, consider Bayesian",
            "Firth's bias correction, exact logistic regression",
            "Lower-order models, Bayesian time series",
            "Cluster-robust standard errors, bias correction",
            "Bayesian SEM, regularized estimation",
            "Exact matching, coarsened exact matching",
            "Synthetic control, permutation tests"
        ]
    })

    st.table(sample_sizes)

    st.subheader("Final Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
                                    **Pre-Analysis**

                                    ✅ Conduct power analysis first

                                    ✅ Focus on fewer, theoretically justified variables

                                    ✅ Consider alternative data sources to augment sample

                                    ✅ Pre-register analyses to avoid p-hacking
                                    """)

    with col2:
        st.markdown("""
                                    **During Analysis**

                                    ✅ Apply appropriate regularization techniques

                                    ✅ Use resampling methods for robust inference

                                    ✅ Consider Bayesian methods with sensible priors

                                    ✅ Conduct sensitivity analyses for all key decisions
                                    """)

    with col3:
        st.markdown("""
                                    **Post-Analysis**

                                    ✅ Report uncertainty honestly and thoroughly

                                    ✅ Focus on effect sizes, not just significance

                                    ✅ Acknowledge limitations of small sample inference

                                    ✅ Consider replication or additional data collection
                                    """)

    st.markdown("""
                                ---

                                Remember that small samples inherently limit the precision and confidence of your conclusions.
                                Focus on robust methods, honest reporting of uncertainty, and cautious interpretation of results.
                                When possible, collect more data or combine multiple sources of information to strengthen inference.
                                """)

st.sidebar.markdown('---')
st.sidebar.info(
    'Created for educational purposes to demonstrate how different econometric methods perform in small samples.by Dr Merwan Roudane')