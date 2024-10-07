import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Set page configuration for a wide layout
st.set_page_config(page_title='Laptop Prices Data Exploration', layout='wide')

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.columns(7)  # Added an extra tab for comparison

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
    with tab7:
        if st.button("Comparison", key="tab_comparison"):  # New Comparison tab
            selected_tab = "Comparison"

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
                with st.expander("Summary", expanded=True):
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
            st.markdown("""The count of features in the dataset reveals the distribution and prevalence of different laptop specifications. For **manufacturers**, a few companies dominate the market, indicating their **popularity** and **brand recognition** among consumers. The types of laptops show a strong preference for categories like **Gaming** and **Ultrabooks**, which reflects current consumer trends and needs. **Screen sizes** predominantly cluster around common dimensions, particularly around **15.6 inches**, suggesting that this size is favored for its balance between usability and portability. **RAM configurations** display considerable diversity, with a significant number of laptops featuring around **8 GB**, catering to both basic and advanced users. **Operating systems** show varied preferences, indicating that consumers are open to different platforms. The presence of **touchscreen options** is notable, reflecting a growing trend towards versatility in laptop design. Advanced features such as **IPS panels** and **Retina displays** are increasingly common, indicating a demand for **high-quality visual experiences**. The data on **primary storage types** shows a shift towards **SSDs**, highlighting consumer preference for **speed** and **efficiency** over traditional **HDDs**. Lastly, the distribution of **CPU and GPU manufacturers** indicates a competitive landscape, with a few key players dominating the market. Overall, the counts of various features illustrate the current landscape of the laptop market, revealing consumer preferences and trends that can inform future product development and marketing strategies.""")

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
            st.write("**Explanation of Key Statistics:**")
            st.write("1. **Minimum Price** shows the cheapest laptop available.")
            st.write("2. **Maximum Price** indicates the highest priced laptop.")
            st.write("3. **Mean Price** gives an average price across all laptops.")
            st.write("4. **Median Price** provides the middle point in price distribution.")
            st.write("5. **Standard Deviation** reflects the price variation among laptops.")

    # Feature Analysis Tab
    elif st.session_state.selected_tab == "Feature Analysis":
        with st.container():
            st.subheader('Feature Analysis')
            feature_analysis_col1, feature_analysis_col2 = st.columns(2)

            with feature_analysis_col1:
                st.subheader('RAM vs Price')
                fig = plt.figure(figsize=(6, 4))  # Adjusted figure size
                sns.boxplot(data=df, x='Ram', y='Price_euros', palette='viridis')
                plt.xlabel('RAM (GB)', fontsize=font_size)
                plt.ylabel('Price (euros)', fontsize=font_size)
                st.pyplot(fig)

            with feature_analysis_col2:
                st.subheader('Screen Size vs Price')
                fig = plt.figure(figsize=(6, 4))  # Adjusted figure size
                sns.boxplot(data=df, x='Inches', y='Price_euros', palette='viridis')
                plt.xlabel('Screen Size (Inches)', fontsize=font_size)
                plt.ylabel('Price (euros)', fontsize=font_size)
                st.pyplot(fig)

            with st.expander("Analysis of Features", expanded=True):
                st.write("""The box plots show how **RAM** and **screen size** affect laptop prices. 
                Higher RAM configurations tend to correlate with higher prices, indicating that consumers are willing to pay more for enhanced performance. 
                The screen size box plot highlights varying prices across different dimensions, revealing trends such as larger screens generally commanding higher prices due to better user experience. 
                These findings suggest that both RAM and screen size are significant factors influencing laptop pricing.""")

    # Multivariate Analysis Tab
    elif st.session_state.selected_tab == "Multivarient Analysis":
        with st.container():
            st.subheader('Multivariate Analysis')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Price vs RAM and Storage Type')
                fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted figure size
                sns.scatterplot(data=df, x='Ram', y='Price_euros', hue='PrimaryStorageType', style='Company', ax=ax, palette='viridis')
                ax.set_title('Price vs RAM with Storage Type', fontsize=title_size)
                ax.set_xlabel('RAM (GB)', fontsize=font_size)
                ax.set_ylabel('Price (euros)', fontsize=font_size)
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                st.pyplot(fig)

            with col2:
                st.subheader('Price vs CPU and GPU')
                fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted figure size
                sns.scatterplot(data=df, x='CPU_model', y='Price_euros', hue='GPU_model', ax=ax, palette='viridis')
                ax.set_title('Price vs CPU and GPU', fontsize=title_size)
                ax.set_xlabel('CPU Model', fontsize=font_size)
                ax.set_ylabel('Price (euros)', fontsize=font_size)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with st.expander("Multivariate Analysis Interpretation", expanded=True):
                st.write("""The scatter plots reveal complex relationships between multiple features. 
                In the first plot, **RAM** and **storage type** significantly impact **price**, with SSDs often correlating with higher prices across various RAM configurations. 
                The second plot illustrates how **CPU** and **GPU** combinations play crucial roles in determining **laptop prices**, reflecting market demand for high-performance components.""")

    # Conclusion Tab
    elif st.session_state.selected_tab == "Conclusion":
        with st.container():
            st.subheader('Conclusion')
            st.write("""Through this data exploration, we discovered significant relationships between laptop specifications and prices. 
            Key findings include the importance of **RAM**, **screen size**, and **storage type** as major price determinants. 
            Understanding these factors can aid consumers in making informed purchasing decisions and manufacturers in optimizing product offerings.""")

    # Comparison Tab
    elif st.session_state.selected_tab == "Comparison":
        with st.container():
            st.subheader('Laptop Comparison')
            search_query = st.text_input("Enter laptop name to search:", "")

            # Filter the DataFrame based on the search query
            filtered_df = df[df['Laptop Name'].str.contains(search_query, case=False)]

            # Display comparison if the user has searched for a laptop
            if not filtered_df.empty:
                if len(filtered_df) > 2:
                    st.warning("Please refine your search to select two laptops for comparison.")
                else:
                    for _, row in filtered_df.iterrows():
                        st.markdown(f"### {row['Laptop Name']}")
                        st.image(row['Image Link'])  # Assuming 'Image Link' column contains the image URLs
                        st.write("**Price:**", row['Price_euros'], "euros")
                        st.write("**RAM:**", row['Ram'], "GB")
                        st.write("**Screen Size:**", row['Inches'], "inches")
                        st.write("**Storage Type:**", row['PrimaryStorageType'])
                        st.write("**CPU Model:**", row['CPU_model'])
                        st.write("**GPU Model:**", row['GPU_model'])
                        st.write("---")  # Separator line
            else:
                st.write("No laptops found. Please try a different search term.")

# Add a footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Aubrey Ocanada</p>", unsafe_allow_html=True)
