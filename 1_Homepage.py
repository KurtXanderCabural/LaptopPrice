import streamlit as st
import io
from PIL import Image
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Set page configuration for a wide layout
st.set_page_config(page_title='Laptop Prices Data Exploration', layout='wide')

# Replace with your actual raw GitHub URL
image_url = "https://raw.githubusercontent.com/KurtXanderCabural/LaptopPrice/main/1.jpg"

# Load the image from the URL
response = requests.get(image_url)
image = Image.open(io.BytesIO(response.content))

# Display the image in Streamlit
st.image(image, caption='Image from GitHub', use_column_width=True)

# If you want to convert the image to bytes (optional)
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='JPEG')
img_byte_arr = img_byte_arr.getvalue()

# Set the CSS for the background
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url("{image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        height: 100vh;
        position: relative;
    }}

    .overlay {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.7);  /* White with transparency */
        z-index: 1;
    }}

    .content {{
        position: relative;
        z-index: 2;  /* Place content above the overlay */
        color: #333;  /* Dark color for better readability */
        padding: 20px;
    }}
    </style>
    <div class="overlay"></div>
    <div class="content">
    """, unsafe_allow_html=True
)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('laptop_prices.csv')
    return df

df = load_data()

# Custom styling for Streamlit components
st.markdown(
    """
    <style>
    .rounded-border {
        border-radius: 15px;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Set up navigation columns
nav_col1, main_col, nav_col2 = st.columns([1, 5, 1]) 

# Initialize selected_tab in session state if not already set
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Home"

# Add left navigation button
with nav_col1:
    st.write("")

# Add right navigation button
with nav_col2:
    st.write("")
    

with main_col:
    st.title('Laptop Prices Data Exploration')

    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            font-weight: bold;
            height: 50px; /* You can adjust this height as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

 # Create a container for the tab buttons
    tab1, tab2, tab3, tab4, tab5, tab6 = st.columns(6)

    with tab1:
        if st.button("Introduction", key="tab_home"):
            selected_tab = "Home"
    with tab2:
        if st.button("Count Analysis", key="tab_count"):
            selected_tab = "Count Analysis"

    with tab3:
        if st.button("Key Statistics", key="tab_statistics"):
            selected_tab = "Key Statistics"

    with tab4:
        if st.button("Feature Analysis", key="tab_laptop_features"):
            selected_tab = "Feature Analysis"

    with tab5:
        if st.button("Multivarient Analysis", key="tab_Multivarient_analysis"):
            selected_tab = "Multivarient Analysis"

    with tab6:
        if st.button("Conclusion", key="tab_Conclusion"):
            selected_tab = "Conclusion"

    # Default tab
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Home"

    # Update the selected tab
    if 'selected_tab' in locals() and selected_tab:
        st.session_state.selected_tab = selected_tab

    font_size = 12  
    title_size = 16  

    # Home Tab
    if st.session_state.selected_tab == "Home":

        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('About the Data!')
                st.write('The dataset used in this analysis is a collection of laptop prices taken from Kaggle. The author, Muhammet Varli, did not specify the collection period but it was updated 4 years ago. The dataset contains details of 1,276 laptops, including technical specifications such as screen size, RAM, storage, CPU, GPU, and weight. The dataset also includes the price of each laptop in euros. The data provides valuable insights into how different features affect laptop pricing and was last updated approximately 4 years ago. The dataset contains both numerical and categorical variables, allowing for descriptive statistics and visual analyses.')
                st.write('This research seeks to determine elements that affect the price and value of a laptop depending on its specs, such as RAM capacity, resolution, and CPU frequency. To achieve this purpose, descriptive statistics and visualizations will be used to discover connections or patterns, as well as device differences, such as budget models vs high-end models, to help us understand what factors affect market value.')
            with col2:
                with st.expander("Summary",expanded=True):
                    st.subheader('Data Preview')
                    st.dataframe(df.head())


    elif st.session_state.selected_tab == "Count Analysis": 
        
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader('Individual Column Count')

                columns_to_plot = ['Company', 'TypeName', 'Inches', 'Ram', 'OS', 'Screen', 'Touchscreen', 'IPSpanel', 'RetinaDisplay', 'PrimaryStorageType', 'CPU_company', 'GPU_company']
                selected_column = st.selectbox("", columns_to_plot, label_visibility="collapsed")
                
                # Create a smaller graph for the individual column count
                fig1, ax1 = plt.subplots(figsize=(6, 3))  # Consistent size for all plots
                sns.countplot(y=selected_column, data=df, palette='viridis', order=df[selected_column].value_counts().index, ax=ax1)
                ax1.set_ylabel('', fontsize=font_size)
                ax1.set_xlabel('Count', fontsize=font_size)
                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                st.pyplot(fig1)
                
            with col2:
                st.subheader('Top Models')

                option = st.selectbox('', ('CPU Models', 'GPU Models'), label_visibility="collapsed")

                # Set up the figure for the top models
                fig2, ax2 = plt.subplots(figsize=(6, 3.15))  # Consistent size for all plots

                if option == 'CPU Models':
                    # Plot CPU distribution
                    sns.countplot(data=df, y='CPU_model', palette='viridis', order=df['CPU_model'].value_counts().head(15).index, ax=ax2)
                    ax2.set_ylabel('CPU Model', fontsize=font_size)
                    ax2.set_xlabel('Count', fontsize=font_size)

                else:
                    # Plot GPU distribution
                    sns.countplot(data=df, y='GPU_model', palette='viridis', order=df['GPU_model'].value_counts().head(15).index, ax=ax2)
                    ax2.set_ylabel('GPU Model', fontsize=font_size)
                    ax2.set_xlabel('Count', fontsize=font_size)

                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.tight_layout()

                # Show the plot in Streamlit
                st.pyplot(fig2)

        with st.expander("Interpretation", expanded=True):
            st.markdown("""
            The count of features in the dataset reveals the distribution and prevalence of different laptop specifications. 
            For **manufacturers**, a few companies dominate the market, indicating their **popularity** and **brand recognition** among consumers. 
            The types of laptops show a strong preference for categories like **Gaming** and **Ultrabooks**, which reflects current consumer trends and needs.
            
            **Screen sizes** predominantly cluster around common dimensions, particularly around **15.6 inches**, suggesting that this size is favored for its balance between usability and portability. 
            **RAM configurations** display considerable diversity, with a significant number of laptops featuring around **8 GB**, catering to both basic and advanced users.
            
            **Operating systems** show varied preferences, indicating that consumers are open to different platforms. The presence of **touchscreen options** is notable, reflecting a growing trend towards versatility in laptop design. 
            Advanced features such as **IPS panels** and **Retina displays** are increasingly common, indicating a demand for **high-quality visual experiences**.
            
            The data on **primary storage types** shows a shift towards **SSDs**, highlighting consumer preference for **speed** and **efficiency** over traditional **HDDs**. 
            Lastly, the distribution of **CPU and GPU manufacturers** indicates a competitive landscape, with a few key players dominating the market. 
            Overall, the counts of various features illustrate the current landscape of the laptop market, revealing consumer preferences and trends that can inform future product development and marketing strategies.
            """)


    # Key Statistics Tab
    elif st.session_state.selected_tab == "Key Statistics":
        st.subheader('Key Statistics')

        col1, col2 = st.columns(2)
        with col1:
                fig, ax = plt.subplots(figsize=(6, 3))  # Smaller graph size
                sns.histplot(df['Price_euros'], bins=60, kde=True, ax=ax)
                ax.set_title('Distribution of Product Prices', fontsize=title_size)
                ax.set_xlabel('Price (euros)', fontsize=font_size)
                ax.set_ylabel('Frequency', fontsize=font_size)
                plt.grid()
                st.pyplot(fig)
        with col2:
                st.write("**Minimum Price:** €174")
                st.write("**Maximum Price:** €6,099")
                st.write("**Mean Price:** €1,134.97")
                st.write("**Median Price:** €989")
                st.write("**Standard Deviation:** €700.75")
                st.write("**Explanation:** The prices vary widely, reflecting both budget and premium models. The mean price of €1,134.97 suggests that, on average, laptops fall within the mid-range segment. The median price of €989 indicates that half of the laptops are priced below this value, showcasing a diverse range of pricing options. The large standard deviation (€700.75) confirms this wide spread, indicating significant price variability between different models.")
        

        col1, col2 = st.columns(2)
        with col2:
            # RAM Distribution
                tabs = st.tabs(["RAM", "Screen Size", "Weight", "Storage", "CPU Frequency"])

                # RAM Distribution Tab
                with tabs[0]:
                    st.write("**Most Common RAM (Mode):** 8GB")
                    st.write("**Mean RAM:** 8.44GB")
                    st.write("**Standard Deviation:** 5.10GB")
                    st.write("**Range:** 2GB to 64GB")
                    st.write("**Explanation:** The mean RAM is approximately 8.44GB, with 8GB being the most common configuration. Higher RAM capacities, such as 16GB and above, are associated with performance-oriented models, which often lead to higher prices. The range of RAM (from 2GB to 64GB) highlights the availability of both entry-level and high-performance laptops. The standard deviation of 5.10GB reflects the variability in RAM capacities, from basic models to advanced, high-end configurations.")

                # Screen Size Distribution Tab
                with tabs[1]:
                    st.write("**Most Common Screen Size:** 15.6 inches")
                    st.write("**Mean Screen Size:** 15.02 inches")
                    st.write("**Standard Deviation:** 1.43 inches")
                    st.write("**Range:** 10.1 inches to 18.4 inches")
                    st.write("**Explanation:** The mean screen size of 15.02 inches reflects the popularity of mid-sized laptops, with 15.6 inches being the most common. The relatively small standard deviation of 1.43 inches indicates that most laptops fall within a standard range, typically between 14 and 16 inches. This suggests that mid-sized laptops dominate the market, with smaller or larger screens being less common.")

                # Weight Distribution Tab
                with tabs[2]:
                    st.write("**Mean Weight:** 2.04 kg")
                    st.write("**Standard Deviation:** 0.67 kg")
                    st.write("**Range:** 0.69 kg to 4.70 kg")
                    st.write("**Explanation:** The average laptop in the dataset weighs around 2.04 kg, with a range from ultra-light models to heavier gaming or workstation laptops. The standard deviation of 0.67 kg indicates moderate variability, with most laptops falling within a common weight range, making them portable for general users.")

                # Storage Distribution Tab
                with tabs[3]:
                    st.write("**Most Common Primary Storage:** 256GB SSD")
                    st.write("**Mean Primary Storage:** 444.52GB")
                    st.write("**Standard Deviation:** 365.54GB")
                    st.write("**Range:** 8GB to 2048GB")
                    st.write("**Explanation:** The data shows a clear preference for SSD storage, with 256GB SSDs being the most common configuration. The mean storage capacity is around 444.52GB, with higher-end laptops offering 512GB or more. The large standard deviation of 365.54GB suggests considerable variability in storage configurations, from budget models with lower storage to high-end devices with more extensive storage options.")

                # CPU Frequency Distribution Tab
                with tabs[4]:
                    st.write("**Mean CPU Frequency:** 2.30 GHz")
                    st.write("**Standard Deviation:** 0.50 GHz")
                    st.write("**Range:** 0.90 GHz to 3.60 GHz")
                    st.write("**Explanation:** The mean CPU frequency of 2.30 GHz suggests that most laptops in the dataset are equipped with mid-range processors, capable of handling general tasks. The standard deviation of 0.50 GHz indicates that there are some laptops with significantly higher or lower processor speeds, depending on the target market (e.g., budget vs. gaming laptops).")
            # Individual Price Distribution
        with col1:

            with st.expander("Summary",expanded=True):
            
                st.write(df.describe())

    # Price by Features Tab
    elif st.session_state.selected_tab == "Feature Analysis":

        # Create a dictionary to map features to their insights
        insights_dict = {
            'Company': "High-end brands like Razer, Microsoft, and Google have higher median prices, indicating their premium positioning in the market. "
                    "On the other hand, brands such as Vero, Chuwi, and LG have a lower median price, showing they target the budget or mid-range market. "
                    "Apple's prices show high consistency, with a narrow range indicating a standard premium pricing strategy.",
            'TypeName': "Workstations, Ultrabooks, and Gaming laptops are typically the most expensive, while Netbooks and Notebooks tend to be cheaper. "
                        "This distribution suggests that high-performance or specialized laptops have a higher market price.",
            'OS': "Laptops with macOS, Windows 7, and Chrome OS are priced at the higher end, with macOS showing consistently premium pricing. "
                "Meanwhile, laptops without any OS or with Linux and Windows 10 S are generally cheaper.",
            'Screen': "Laptops with 4K Ultra HD and Quad HD+ screens have the highest median prices, while those with Standard screens are on the lower end. "
                    "This suggests that higher-resolution screens significantly contribute to the overall laptop cost."
        }

        # Dummy insights for numeric features
        numeric_insights_dict = {
            'Ram': "Higher RAM usually leads to better multitasking capabilities and overall performance.",
            'Weight': "Laptops with lower weight are generally more portable and favored by users who travel frequently.",
            'CPU_freq': "Higher CPU frequencies indicate better processing power, affecting the laptop's speed and efficiency.",
            'PrimaryStorage': "Laptops with larger storage capacities can store more data, impacting performance based on storage type."
        }

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Price vs Categorical Features')
            selected_feature = st.selectbox("", list(insights_dict.keys()), label_visibility="collapsed")

            # Create the boxplot immediately after feature selection
            fig, ax = plt.subplots(figsize=(8, 4))  # Slightly larger for better readability
            sns.boxplot(data=df, x=selected_feature, y='Price_euros', palette='Spectral', ax=ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylabel('Price (euros)', fontsize=font_size)  # Customize ylabel font size
            ax.set_xlabel('')
            # Display the boxplot
            st.pyplot(fig)

            # Immediately display insights related to the selected feature
            st.write(f"**{selected_feature}:** {insights_dict[selected_feature]}")

        with col2:
            st.subheader('Price vs Numeric Features')
            num_features = ['Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
            selected_numeric_feature = st.selectbox("Select a feature", num_features, label_visibility="collapsed")

            # Create the scatter plot with Weight vs Price_euros if Weight is selected
            fig, ax = plt.subplots(figsize=(10, 6))
            if selected_numeric_feature == 'Weight':
                sns.scatterplot(x='Weight', y='Price_euros', data=df, hue='TypeName', alpha=0.7, ax=ax)
                ax.set_xlabel('Weight (kg)', fontsize=font_size)  # Customize xlabel font size
                ax.set_ylabel('Price (euros)', fontsize=font_size)  # Customize ylabel font size
            else:
                # Scatter plot for other selected features
                sns.scatterplot(data=df, x=selected_numeric_feature, y='Price_euros', hue='TypeName', alpha=0.7, ax=ax)
                ax.set_xlabel(selected_numeric_feature, fontsize=font_size)  # Customize xlabel font size
                ax.set_ylabel('Price (euros)', fontsize=font_size)  # Customize ylabel font size

            ax.grid()
            ax.legend()
            st.pyplot(fig)

            # Immediately display dummy insights related to the selected numeric feature
            st.write(f"**{selected_numeric_feature}:** {numeric_insights_dict[selected_numeric_feature]}")



    # Price vs Numeric Features Tab
    elif st.session_state.selected_tab == "Multivarient Analysis": 
        
        # Adjusting column width: 3 parts for col1 and 1 part for col2
        col1, col2 = st.columns([3, 1])

        # Count Plot in first column
        with col1:

            st.subheader('Correlation Heatmap')
            # Label encoding categorical features for correlation analysis
            label_encoder = LabelEncoder()

            object_columns = df.select_dtypes(include=['object']).columns

            for col in object_columns:
                df[col] = label_encoder.fit_transform(df[col])
                    
            corr = df.select_dtypes(include=[np.number]).corr()

            fig, ax = plt.subplots(figsize=(9, 4))  # Smaller heatmap size
            sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, ax=ax)  # Removed annot=True
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)  # X-tick labels
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size)  # Y-tick labels
                
            st.pyplot(fig)

        # Price Distribution in second column
        with col2:
            st.subheader('Summary')

            correlations = df.corr()['Price_euros'].abs().sort_values(ascending=False)
            st.write(correlations)

    elif st.session_state.selected_tab == "Conclusion":

        # Correlation Analysis Section
        st.subheader("Correlation Analysis")
        
        with st.expander("Key Relationships between Features and Laptop Prices", expanded=True):
            st.markdown("""
            - **Price and RAM**: The strongest positive correlation with laptop price is RAM (**0.74**), suggesting that laptops with more RAM tend to be significantly more expensive.
            - **Screen Size**: Both 'Inches' (**0.39**) and 'ScreenW' (**0.55**) have a moderate positive correlation with price. This indicates that laptops with larger screens tend to be more costly.
            - **CPU and GPU Characteristics**: The CPU frequency (**0.43**) and GPU company (**0.48**) also show a moderate correlation with laptop prices, signifying that more advanced CPUs and GPUs contribute to a higher price.
            - **Weight**: There is a moderate positive correlation (**0.39**) between laptop weight and price, likely because more powerful components and larger screens make laptops heavier.
            - **Primary and Secondary Storage**: A weak positive correlation exists between Primary Storage (**0.18**) and price, indicating that laptops with larger primary storage capacities are slightly more expensive. Secondary Storage Type has a negative correlation (**-0.52**), meaning that certain types of secondary storage (likely HDDs) are associated with cheaper laptops.
            - **Retina Display and Touchscreen**: Both features have weak correlations with price, suggesting they have some influence but are not the primary determinants.
            
            In summary, RAM, screen size, CPU/GPU capabilities, and weight are the most influential factors driving laptop prices in this dataset, while other attributes have lesser but noticeable effects.
            """)

        # Conclusion Section
        st.subheader("Final Insights")
        
        with st.expander("Summary of Key Factors", expanded=True):
            st.markdown("""
            From analyzing the data in our dataset, we conclude that **RAM** emerges as the most significant determinant for laptop prices. Laptop prices with higher RAM provide a crucial role in system performance, supporting the notion that more RAM generally leads to higher prices. Factors like screen specifications, storage types, and primary storage only came in second, third, and fourth respectively.
            
            - **RAM** (**0.740287**): Emerges as the most significant determinant for laptop prices. Higher RAM provides a crucial role in system performance, supporting the notion that more RAM generally leads to higher prices.
            - **Screen Width** (**0.552491**) and **Screen Height** (**0.548529**): These screen specifications show a moderate positive correlation with price, indicating that display quality is an important factor in pricing.
            - **Primary Storage Type** (**0.503655**): Has a moderate correlation with price, suggesting that the type of storage (likely SSD vs HDD) plays a notable role in determining a laptop's cost.
            - **CPU Model** (**0.473860**) and **CPU Frequency** (**0.428847**): Both show a moderate correlation with price, indicating that processor specifications are important, but not as crucial as RAM or screen specs in this dataset.
            - **Screen** (**0.403834**): This general screen factor (possibly referring to overall screen size or quality) shows the weakest correlation among the listed factors, but still has a moderate positive influence on price.
            """)

