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
        if st.button("Multivariate Analysis", key="tab_Multivariate_analysis"):
            selected_tab = "Multivariate Analysis"
    with tab6:
        if st.button("Conclusion", key="tab_Conclusion"):
            selected_tab = "Conclusion"
    with tab7:
        if st.button("Comparison", key="tab_comparison"):  # New Comparison tab
            selected_tab = "Comparison"

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
            st.write("3. **Mean Price** provides the average laptop price.")
            st.write("4. **Median Price** shows the middle point of the dataset.")
            st.write("5. **Standard Deviation** reveals the variability in laptop prices.")

    # Feature Analysis Tab
    elif st.session_state.selected_tab == "Feature Analysis":
        st.subheader('Feature Analysis')
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Distribution of RAM and Price")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(data=df, x='Ram', y='Price_euros', ax=ax)
            ax.set_title('Price vs RAM', fontsize=title_size)
            ax.set_xlabel('RAM (GB)', fontsize=font_size)
            ax.set_ylabel('Price (euros)', fontsize=font_size)
            st.pyplot(fig)

        with col2:
            st.write("### Distribution of Storage Type and Price")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(data=df, x='PrimaryStorageType', y='Price_euros', ax=ax)
            ax.set_title('Price vs Storage Type', fontsize=title_size)
            ax.set_xlabel('Storage Type', fontsize=font_size)
            ax.set_ylabel('Price (euros)', fontsize=font_size)
            st.pyplot(fig)

        with st.expander("Interpretation", expanded=True):
            st.markdown("""The analysis of features and their relationship with price reveals key insights. The **box plot for RAM** indicates that laptops with higher RAM tend to have a higher price, suggesting that RAM is a critical factor in determining a laptop's cost. Additionally, it shows that laptops with 16GB RAM and higher significantly surpass the average price, indicating a premium segment in the market. 

            The **box plot for storage types** indicates a clear price differentiation based on storage technology. Laptops with **SSD** storage are generally more expensive than their **HDD** counterparts. This analysis emphasizes the importance of RAM and storage type as significant contributors to laptop pricing."""

    # Multivariate Analysis Tab
    elif st.session_state.selected_tab == "Multivariate Analysis":
        st.subheader('Multivariate Analysis')
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 3)) 
            sns.scatterplot(data=df, x='Inches', y='Price_euros', hue='TypeName', palette='viridis', ax=ax)
            ax.set_title('Price vs Screen Size by Laptop Type', fontsize=title_size)
            ax.set_xlabel('Screen Size (inches)', fontsize=font_size)
            ax.set_ylabel('Price (euros)', fontsize=font_size)
            plt.grid()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(data=df, x='Company', y='Price_euros', ax=ax)
            ax.set_title('Price Distribution by Company', fontsize=title_size)
            ax.set_xlabel('Company', fontsize=font_size)
            ax.set_ylabel('Price (euros)', fontsize=font_size)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)

        with st.expander("Interpretation", expanded=True):
            st.markdown("""The **scatter plot** indicates that as screen size increases, there tends to be a moderate increase in price. However, the price variability also suggests that screen size alone does not solely determine laptop pricing. Different **types of laptops** also influence price significantly, with **gaming laptops** often positioned at the higher end of the price spectrum.

            The **box plot** for company price distribution shows that certain brands consistently charge more for their laptops, highlighting brand positioning and market strategy. Companies like **Apple** command higher prices, likely due to their strong brand equity and premium features, while others like **HP** or **Acer** show a broader range of prices, indicating they cater to both budget and premium segments.""")

    # Conclusion Tab
    elif st.session_state.selected_tab == "Conclusion":
        st.subheader('Conclusion')
        st.write("### Key Findings")
        st.write("- The dataset reveals a diverse range of laptop prices, influenced by various specifications such as RAM, storage type, and brand.")
        st.write("- Higher RAM and SSD storage generally correlate with higher prices.")
        st.write("- Consumer preferences lean towards laptops that balance performance with price, emphasizing the importance of features like screen size and brand reputation.")

        st.write("### Recommendations")
        st.write("- Consumers should consider their specific needs when selecting a laptop, focusing on specifications that align with their usage patterns.")
        st.write("- Brands may benefit from highlighting the features that contribute most to price differentiation in their marketing strategies.")

    # Comparison Tab
    elif st.session_state.selected_tab == "Comparison":
        st.subheader('Laptop Comparison')

        # Load the comparison data from CSV
        comparison_df = pd.read_csv('laptop_prices.csv')

        # User input for laptop selection
        laptop1 = st.selectbox("Select First Laptop", comparison_df['Laptop_Name'].unique())
        laptop2 = st.selectbox("Select Second Laptop", comparison_df['Laptop_Name'].unique())

        # Filter data for the selected laptops
        laptop1_data = comparison_df[comparison_df['Laptop_Name'] == laptop1].iloc[0]
        laptop2_data = comparison_df[comparison_df['Laptop_Name'] == laptop2].iloc[0]

        # Display selected laptops' information
        st.markdown("### Comparison Results")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(laptop1)
            st.image(laptop1_data['Image'], width=200)  # Display image for Laptop 1
            st.write(f"**Price:** €{laptop1_data['Price_euros']}")
            st.write(f"**RAM:** {laptop1_data['Ram']} GB")
            st.write(f"**Storage:** {laptop1_data['PrimaryStorageType']}")
            st.write(f"**CPU:** {laptop1_data['CPU_model']}")
            st.write(f"**GPU:** {laptop1_data['GPU_model']}")
            st.write(f"**Screen Size:** {laptop1_data['Inches']} inches")
            st.write(f"**Weight:** {laptop1_data['Weight']}")
        
        with col2:
            st.subheader(laptop2)
            st.image(laptop2_data['Image'], width=200)  # Display image for Laptop 2
            st.write(f"**Price:** €{laptop2_data['Price_euros']}")
            st.write(f"**RAM:** {laptop2_data['Ram']} GB")
            st.write(f"**Storage:** {laptop2_data['PrimaryStorageType']}")
            st.write(f"**CPU:** {laptop2_data['CPU_model']}")
            st.write(f"**GPU:** {laptop2_data['GPU_model']}")
            st.write(f"**Screen Size:** {laptop2_data['Inches']} inches")
            st.write(f"**Weight:** {laptop2_data['Weight']}")

        # Comparison insights
        st.write("### Insights")
        if laptop1_data['Price_euros'] < laptop2_data['Price_euros']:
            st.write(f"{laptop1} is cheaper than {laptop2}.")
        elif laptop1_data['Price_euros'] > laptop2_data['Price_euros']:
            st.write(f"{laptop2} is cheaper than {laptop1}.")
        else:
            st.write(f"Both laptops have the same price.")

        if laptop1_data['Ram'] > laptop2_data['Ram']:
            st.write(f"{laptop1} has more RAM than {laptop2}.")
        elif laptop1_data['Ram'] < laptop2_data['Ram']:
            st.write(f"{laptop2} has more RAM than {laptop1}.")
        else:
            st.write(f"Both laptops have the same RAM.")

        if laptop1_data['PrimaryStorageType'] == laptop2_data['PrimaryStorageType']:
            st.write(f"Both laptops have the same type of storage: {laptop1_data['PrimaryStorageType']}.")
        else:
            st.write(f"{laptop1} has {laptop1_data['PrimaryStorageType']} storage while {laptop2} has {laptop2_data['PrimaryStorageType']}.")

        # Add more insights as necessary...

# Add footer for Streamlit app
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)
