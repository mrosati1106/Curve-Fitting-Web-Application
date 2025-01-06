import streamlit as st
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


st.set_page_config(page_title="Matteo Rosati Curve Fit", layout='wide')
st.markdown("# DataSculpt")
st.caption("Matteo Rosati | Streamlit Webapp")
st.write("Instructions on how to use the app below.")


def histogram(x, line_color, bg_color, hist_color, bins):
    fig, ax = plt.subplots()

    mean, std_dev = norm.fit(x)
    range_given_x = np.max(x)-np.min(x)
    min_given_x = np.min(x)
    max_given_x = np.max(x)
    graph_x = np.linspace(int(min_given_x-.5*range_given_x),
                          int(max_given_x+.5*range_given_x), 1000)

    counts, bin_edges, patches = ax.hist(x, bins=bins, edgecolor='black', alpha=0.6,
                                         label='Histogram', color=hist_color)
    bin_width = bin_edges[1] - bin_edges[0]
    y = norm.pdf(graph_x, mean, std_dev) * bin_width * len(x)
    ax.plot(graph_x, y, label=f'Fit: μ={mean:.2f}, σ={
            std_dev:.2f}', color=line_color)
    ax.set_facecolor(bg_color)
    mean = round(mean, 3)
    std_dev = round(std_dev, 3)
    text = fr"""f(x) = \frac{{1}}{{{
        std_dev} \sqrt{{2\pi}}}} \cdot e^{{-\frac{{1}}{{2}} \left( \frac{{x - {mean}}}{{{std_dev}}} \right)^2 }}"""
    text2 = (rf"$f(x) = \frac{{1}}{{{
             std_dev} \sqrt{{2\pi}}}} \cdot e^{{-\frac{{1}}{{2}} \left(\frac{{x - {mean}}}{{{std_dev}}}\right)^2}}$")
    plt.figtext(
        x=.5,
        y=.0001,
        ha='center',
        s=minus_check(text2),
        color=line_color,
        bbox={"boxstyle": "round",
              "edgecolor": line_color, "facecolor": "none"}
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf_values = norm.pdf(bin_centers, mean, std_dev) * bin_width * len(x)
    errors = np.abs(counts - pdf_values)

    return (fig, ax, text, errors)


def cosine(x, a, b, c, d):
    f = a*np.cos(b*(x-c))+d
    return f


def sine(x, a, b, c, d):
    f = a*np.sin(b*(x-c))+d
    return f


def arrhenius(x, A, E):
    f = A*np.exp(-E/(8.314*x))
    return f


def exponential(x, a, b, c, d):
    f = a*np.exp(b*(x-c))+d
    return f


def nat_log(x, a,  b, c, d):
    f = a*np.log(b*(x-c))+d
    return f


def log_10(x, a, b, c, d):
    f = a*np.log10(b*(x-c))+d
    return f


def log_2(x, a, b, c, d):
    f = a*np.log2(b*(x-c))+d
    return f


def normal_dist(x, sigma, mu):
    f = (1 / (sigma * (2 * np.pi)**0.5)) * \
        np.exp((-1 / 2) * ((x - mu) / sigma)**2)
    return f


def polyn(coefficients, init_degree):
    coefficients = np.round(coefficients, decimals=3)
    coefficients = np.where(np.abs(coefficients) < 1e-3, 0, coefficients)
    f = 'f(x) = '
    degree = init_degree
    first = True
    for coeff in coefficients:
        if coeff != 0:
            if first == False:
                f += " + "
            if degree == 0:
                f += str(coeff)
            elif degree > 1:
                f += str(coeff)+'x^'+str(degree)
            elif degree == 1:
                f += str(coeff)+'x'

            first = False
        degree -= 1
        f = f.replace('+ -', ' - ')
    return f


def minus_check(string):
    string = string.replace('+ -', ' - ')
    string = string.replace('- +', ' - ')
    string = string.replace('- -', ' + ')
    string = string.replace('--', '')

    return string


def equation_to_latex(equation: str) -> str:
    import re
    equation = re.sub(r"x\^(\d+)", r"x^{\1}", equation)
    latex_equation = f"${equation}$"
    return latex_equation


def generate_graph(type_func, x, y, point_color, line_color, bg_color):
    covariance = None
    min_given_x = np.min(x)
    max_given_x = np.max(x)
    range_given_x = np.max(x)-np.min(x)
    graph_x = np.linspace(int(min_given_x-.5*range_given_x),
                          int(max_given_x+.5*range_given_x), 1000)
    if type_func == 'Polynomial':
        coeff = np.polyfit(x, y, degree)
        graph_y = np.polyval(coeff, graph_x)
        resid_y = np.polyval(coeff, x)

        function_text = equation_to_latex(polyn(coeff, degree))

    elif type_func == 'Cosine':
        coeff, covariance = curve_fit(
            cosine, xdata=x, ydata=y, p0=(1,  1, 0, 0))
        a, b, c, d = coeff
        graph_y = a*np.cos(b*(graph_x-c))+d
        resid_y = a*np.cos(b*(x-c))+d

        a = round(a, 3)
        b = round(b, 3)
        c = round(c, 3)
        d = round(d, 3)

        function_text = equation_to_latex(minus_check('f(x) = '+str(a) +
                                          "cos(x - " + str(c) + ') + ' + str(d)))

    elif type_func == 'Sine':
        coeff, covariance = curve_fit(
            sine, xdata=x, ydata=y, p0=(1, 1, 0, 0), maxfev=1000000)
        a, b, c, d = coeff
        graph_y = a*np.sin(b*(graph_x-c))+d
        resid_y = a*np.sin(b*(x-c))+d

        a = round(a, 3)
        b = round(b, 3)
        c = round(c, 3)
        d = round(d, 3)

        function_text = equation_to_latex(minus_check('f(x) = '+str(a) +
                                          "sin("+str(b)+"(x - " + str(c) + ')) + ' + str(d)))

    elif type_func == 'Arrhenius':
        coeff, covariance = curve_fit(
            arrhenius, xdata=x, ydata=y, p0=(1, 1))
        A, E = coeff
        graph_y = A*np.exp(-E/(8.314*graph_x))
        resid_y = A*np.exp(-E/(8.314*x))

        A = round(A, 3)
        E = round(E, 3)
        function_text = rf"$f(x) = {A} \cdot e^{{-\frac{{{E}}}{{R x}}}}$"

    elif type_func == 'Exponential':
        coeff, covariance = curve_fit(
            exponential, xdata=x, ydata=y, p0=(1, 1, 0, 0))
        a, b, c, d = coeff
        graph_y = a*np.exp(b*(graph_x-c))+d
        resid_y = a*np.exp(b*(x-c))+d

        a = round(a, 3)
        b = round(b, 3)
        c = round(c, 3)
        d = round(d, 3)
        function_text = f"${a} \\cdot e^{{{b}(x - {c})}} + {d}$"

    elif type_func == 'Logarithmic':
        if base == 'e':
            log_function = nat_log
        elif base == 2:
            log_function = log_2
        elif base == 10:
            log_function = log_10

        coeff, covariance = curve_fit(
            log_function, xdata=x, ydata=y, p0=(1, 1, 0, 0))
        a, b, c, d = coeff
        if base == 'e':
            graph_y = a*np.log(b*(graph_x-c))+d
            resid_y = a*np.log(b*(x-c))+d
        elif base == 2:
            graph_y = a*np.log2(b*(graph_x-c))+d
            resid_y = a*np.log2(b*(x-c))+d
        elif base == 10:
            graph_y = a*np.log10(b*(graph_x-c))+d
            resid_y = a*np.log10(b*(x-c))+d

        a = round(a, 3)
        b = round(b, 3)
        c = round(c, 3)
        d = round(d, 3)

        if base == "e":
            function_text = minus_check(
                f"${a} \\cdot \\ln({b}(x - {c})) + {d}$")
        else:
            function_text = minus_check(
                f"${a} \\cdot \\log_{{{base}}}({b}(x - {c})) + {d}$")

    fig, ax = plt.subplots()
    ax.scatter(x, y, color=point_color, label='Custom Points', s=point_size)
    ax.plot(graph_x, graph_y, color=line_color)
    plt.figtext(
        x=.5,
        y=.0001,
        ha='center',
        s=minus_check(function_text),
        color=line_color,
        bbox={"boxstyle": "round",
              "edgecolor": line_color, "facecolor": "none"}
    )
    residuals = y-resid_y
    ax.set_facecolor(bg_color)
    if covariance is not None:
        dev = np.sqrt(np.diagonal(covariance))
    else:
        dev = np.std(residuals)
    return (fig, ax, dev, residuals, function_text)


fig, ax = plt.subplots()
col1, col2 = st.columns([1, 3])
with col1:
    type_func = st.radio("Select Function type:", [
        "Polynomial", "Cosine", "Sine", "Arrhenius", "Exponential", "Logarithmic"])
st.write("---")
col3, col4 = st.columns([2, 1])
with col2:

    tab1, tab2 = st.tabs(["Enter Custom Data", "Upload CSV Data"])

    with tab1:
        st.markdown("Enter x/y data:")
        initial_data = pd.DataFrame([
            {'x': 0.0, 'y': 0.0}
        ])
        data_table = st.data_editor(
            initial_data, num_rows="dynamic", use_container_width=True)
        table_x = data_table["x"].to_numpy()
        table_y = data_table["y"].to_numpy()

    with tab2:
        file = st.file_uploader('Upload x/y data:', type=["csv"])
        if file is not None:
            try:
                df = pd.read_csv(file)
                file_x = df.iloc[:, 0].to_numpy()
                file_y = df.iloc[:, 1].to_numpy()
            except:
                st.error(
                    "Ensure CSV file has x and y data, even when plotting a histogram.")

    if type_func == 'Polynomial':
        degree = st.number_input("Enter Degree of Polynomial:", 0, 10)
    elif type_func == 'Logarithmic':
        base = st.radio("Select a base:", ['e', 10, 2], horizontal=True)

    col8, col5, col6, col7 = st.columns([1, 1, 1, 2])
    with col5:
        function_color = st.color_picker(
            "Curve Colour:", value="#2D82B7", key="function_color")
    with col6:
        data_color = st.color_picker(
            "Data Point Colour:", value="#E03616", key="data_color")
    with col7:
        point_size = st.slider("Data Point Size", value=5,
                               min_value=1, max_value=50)
    with col8:
        background_color = st.color_picker(
            "Background Colour:", value='#FFFFFF', key="background_color")

    with tab1:
        colbutton1, colbutton2, colbutton7, colbutton3 = st.columns([
                                                                    .85, 1, .75, 1.33])
        with colbutton1:
            if st.button("Generate Graph from Table"):
                try:
                    fig, ax, deviation, residuals, fx_text = generate_graph(
                        type_func, table_x, table_y, data_color, function_color, background_color)

                    with col4:
                        avg_error = np.mean(np.abs(residuals))
                        max_error = np.max(np.abs(residuals))
                        st.markdown(
                            "<h4 style='text-decoration: underline;'>Graph Data</h4>", unsafe_allow_html=True)
                        st.write("Mean Value: " +
                                 str(round(np.mean(table_x), 3)))
                        st.write("Standard Deviation: " +
                                 str(round(np.std(table_x), 3)))
                        st.write("Average Error: ± "+str(round(avg_error, 3)))
                        st.write("Maximum Error: ± "+str(round(max_error, 3)))
                        st.write(fx_text)
                except:
                    st.error(
                        "Error generating graph. Ensure ample data is imported to fit curve.")
        with colbutton7:
            hist_color = st.color_picker(
                "Histogram Box Colour:", value='#FF9999', key="hist_color")
        with colbutton3:
            bins = st.slider("Number of Bins", value=10,
                             min_value=1, max_value=50, key='bin')
        with colbutton2:
            if st.button("Generate Histogram from Table"):
                try:
                    fig, ax, fx_text, error = histogram(
                        table_x, function_color, background_color, hist_color, bins)
                    with col4:
                        st.markdown(
                            "<h4 style='text-decoration: underline;'>Graph Data</h4>", unsafe_allow_html=True)
                        st.write("Mean Value: " +
                                 str(round(np.mean(table_x), 3)))
                        st.write("Standard Deviation: " +
                                 str(round(np.std(table_x), 3)))
                        st.write("Average Error: ± " +
                                 str(round(np.mean(error), 3)))
                        st.write("Maximum Error: ± " +
                                 str(round(np.max(error), 3)))

                        st.latex(minus_check(fx_text))
                except:
                    st.error(
                        "Error generating graph. Ensure ample data is imported to fit curve.")

    with tab2:
        colbutton4, colbutton5, colbutton8, colbutton6 = st.columns([
                                                                    .85, 1, .75, 1.33])
        with colbutton4:
            if st.button("Generate Graph from File"):
                if file == None:
                    st.error("No file has been uploaded.")
                else:
                    try:
                        fig, ax, deviation, residuals, fx_text = generate_graph(
                            type_func, file_x, file_y, data_color, function_color, background_color)
                        with col4:
                            avg_error = np.mean(np.abs(residuals))
                            max_error = np.max(np.abs(residuals))
                            st.markdown(
                                "<h4 style='text-decoration: underline;'>Graph Data</h4>", unsafe_allow_html=True)
                            st.write("Average Error: ± " +
                                     str(round(avg_error, 3)))
                            st.write("Maximum Error: ± " +
                                     str(round(max_error, 3)))
                            st.write(fx_text)
                    except:
                        st.error(
                            "CSV file not formatted properly. Try again. Ensure ample data is imported to fit curve.")
        with colbutton8:
            hist_color2 = st.color_picker(
                "Histogram Box Colour:", value='#FF9999', key="hist_color2")
        with colbutton6:
            bins2 = st.slider("Number of Bins", value=10,
                              min_value=1, max_value=50, key='bin2')
        with colbutton5:
            if st.button("Generate Histogram from File", key='file_hist_button'):
                if file == None:
                    st.error("No file has been uploaded.")
                else:
                    try:
                        fig, ax, fx_text, error = histogram(file_x, function_color,
                                                            background_color, hist_color2, bins2)
                        with col4:
                            st.markdown(
                                "<h4 style='text-decoration: underline;'>Graph Data</h4>", unsafe_allow_html=True)
                            st.write("Mean Value: " +
                                     str(round(np.mean(file_x), 3)))
                            st.write("Standard Deviation: " +
                                     str(round(np.std(file_x), 3)))
                            st.write("Average Error: ± " +
                                     str(round(np.mean(error), 3)))
                            st.write("Maximum Error: ± " +
                                     str(round(np.max(error), 3)))
                            st.latex(minus_check(fx_text))

                    except:
                        st.error(
                            "CSV file not formatted properly. Try again. Ensure histogram data is still in x, y format.")


with col3:
    graph = st.pyplot(fig)
st.divider()
st.header("Instructions for Using DataSculpt:")
st.subheader("Entering Data:")
st.write("You can input your data in two ways:")
st.markdown('- **Manual Entry:** Use the provided text field in the "Enter Custom Data" tab to input x and y values manually.')
st.markdown('- **CSV Upload:** Switch to the "Upload CSV Data" tab, drag and drop your .csv file, or browse your device to upload a file. Ensure the file is in the correct x,y format.')
st.subheader("Selecting Curve Types:")
st.markdown("- Choose a curve type from the provided options, such as Polynomial, Sine, Exponential, and others. Select one that fits your data characteristics.")
st.subheader("Polynomial Fitting:")
st.markdown('- If using a polynomial fit, specify the degree of the polynomial in the input field labeled "Enter Degree of Polynomial."')
st.subheader("Customizing Visualizations:")
st.write("Adjust the following visualization settings as needed:")
st.markdown(
    '- Curve Colour, Data Point Colour, and Background Colour for the graph.')
st.markdown(
    '- Data Point Size using the slider to make data points more or less prominent.')
st.markdown('- If generating a histogram, set the Histogram Box Colour and Number of Bins for better data distribution visualization. **Be sure to still include y data. Although y data is not considered when plotting histograms, the program accepts only x/y value pairs as a csv file.**')
st.subheader("Generating Outputs:")
st.write("Use the appropriate buttons to create visual outputs:")
st.markdown(
    '- **Generate Graph:** Displays a graph of your data along with the fitted curve.')
st.markdown('- **Generate Histogram:** Creates a histogram of the x data.')
st.subheader("Interpreting Results:")
st.markdown('- View the fitted equation displayed below the graph.')
st.markdown('- Key metrics like Average Error and Maximum Error will appear in the output section to evaluate the quality of the fit""")')
