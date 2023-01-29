import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go

import io
import requests

from PIL import Image
from streamlit_extras.badges import badge

# Import sklearn methods/tools
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, silhouette_score, calinski_harabasz_score

# Import all sklearn algorithms used
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture


def main():
    col1, col2, col3 = st.columns([0.05, 0.265, 0.035])
    
    with col1:
        url = 'https://github.com/tsu2000/customer_segment/raw/main/images/pie.png'
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        st.image(img, output_format = 'png')

    with col2:
        st.title('&nbsp; Customer Segmentation')

    with col3:
        badge(type = 'github', name = 'tsu2000/customer_segment', url = 'https://github.com/tsu2000/customer_segment')

    st.markdown('### 🏪 &nbsp; Customer Segmentation Analysis Machine Learning App')
    st.markdown('This web application aims to explore the effectiveness of various clustering models for the selected customer dataset with different features. The original source of the data can be found [**here**](<https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis>).')

    # Initialise dataframe
    url = 'https://raw.githubusercontent.com/tsu2000/customer_segment/main/customers.csv'
    df = pd.read_csv(url)

    # Checking for null/infinite values and replacing with mean values:
    # df[df.isin([np.NaN, -np.Inf, np.Inf]).any(axis=1)] # Checks if any data has missing/infinite values
    df['Income'].fillna(df['Income'].mean(), inplace = True)

    # Manual replacement of data values:
    df['Marital_Status'].replace({'Widow': 'Widowed',
                                  'Together': 'Married',
                                  'Alone': 'Single',
                                  'Absurd': 'Single',
                                  'YOLO': 'Single'}, 
                                  inplace = True)

    options = st.selectbox('Select a feature/machine learning model:', ['Exploratory Data Analysis',
                                                                        'K Means Clustering',
                                                                        'DBSCAN (Density-Based Spatial Clustering)',
                                                                        'Hierarchical Clustering',
                                                                        'Gaussian Mixture Model (GMM)',
                                                                        'Spectral Clustering'])

    st.markdown('---')

    if options == 'Exploratory Data Analysis':
        eda(data = df)
    elif options == 'K Means Clustering':
        k_means_model(data = df)
    elif options == 'DBSCAN (Density-Based Spatial Clustering)':
        dbscan_model(data = df)
    elif options == 'Hierarchical Clustering':
        hierarch_model(data = df)
    elif options == 'Gaussian Mixture Model (GMM)':
        gaussian_model(data = df)
    elif options == 'Spectral Clustering':
        spectral_model(data = df)


def scaled_processing(data):
    # Drop columns here
    data = data.drop(['ID', 'Dt_Customer'], axis = 1)

    # Preprocessing and scaling data
    ohe = OneHotEncoder(drop = 'first')
    mms = MinMaxScaler()

    # Make column transformer
    ct = make_column_transformer(
        (ohe, ['Education', 'Marital_Status']),
        (mms, [data.columns[0]]),
        (mms, data.columns[3:26]),
        remainder = 'passthrough'
    )

    prep_df = ct.fit_transform(data)
    df = pd.DataFrame(columns = ct.get_feature_names_out(), data = prep_df)

    return df


#@st.cache(suppress_st_warning = True, allow_output_mutation = True)
def clustering_report(data, true_labels, cluster_labels):
    df = pd.DataFrame(
        index = ['Homogeneity', 
                 'Completeness', 
                 'V-Measure', 
                 'Adjusted Rand Index', 
                 'Silhoutte Score', 
                 'Calinski-Harabasz Index Score'],
        data = [homogeneity_score(true_labels, cluster_labels), 
                completeness_score(true_labels, cluster_labels),
                v_measure_score(true_labels, cluster_labels), 
                adjusted_rand_score(true_labels, cluster_labels), 
                silhouette_score(data, cluster_labels), 
                calinski_harabasz_score(data, cluster_labels)],
        columns = ['Values']
    ).round(4)

    fig = go.Figure(data = [go.Table(columnwidth = [4, 1.75],
                                header = dict(values = ['<b>Clustering Metric<b>', '<b>Score/Value<b>'],
                                                fill_color = 'navy',
                                                line_color = 'darkslategray',
                                                font = dict(color = 'white', size = 16)),
                                    cells = dict(values = [df.index, df.values], 
                                                fill_color = ['lightblue']*6,
                                                line_color = 'darkslategray',
                                                align = ['right', 'left'],
                                                font = dict(color = [['navy']*6, 
                                                                     ['navy']*6], 
                                                            size = [14, 14]),
                                                height = 25))])
    fig.update_layout(height = 200, width = 700, margin = dict(l = 5, r = 5, t = 5, b = 5))
    return st.plotly_chart(fig, use_container_width = True)


def eda(data):
    st.markdown('## 🔎 &nbsp; Exploratory Data Analysis (EDA)')

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Selection')
        df_type = st.selectbox('Select type of DataFrame to be displayed:', ['Initial DataFrame', 'Modified DataFrame'])

        st.markdown('### Initial DataFrame Info:')
        st.markdown('- Pandas DataFrame as extracted from `.csv` file on GitHub.')

        st.markdown('### Modified DataFrame Info:')
        st.markdown("- All null values for 'Income' have been replaced with the mean income.")
        st.markdown("- Customer ID and Date of customers' registration with company columns have been dropped.")
        st.markdown("- One-Hot Encoding applied to categorical variables of 'Education', 'Marital_Status' with one column from each category dropped to avoid the 'dummy variable trap'.")
        st.markdown("- MinMax Scaler applied to the rest of the numerical variables.")
        st.markdown("- No changes made to the dependent variable 'Response' at all.")

    if df_type == 'Initial DataFrame':
        st.markdown('### Initial DataFrame:')
        st.dataframe(data, use_container_width = True)
        st.write(f'Shape of data:', data.shape)

    elif df_type == 'Modified DataFrame':
        st.markdown('### Modified DataFrame:')
        mod_data = scaled_processing(data)
        st.dataframe(mod_data, use_container_width = True)
        st.write(f'Shape of data:', mod_data.shape)

    st.markdown('### Summary Statistics:')
    st.dataframe(data.describe(), use_container_width = True)

    data2 = data.copy()
    data2 = data2.drop(['Z_CostContact', 'Z_Revenue'], axis = 1)

    # Factorise categorical variables
    data2['Education'] = pd.factorize(data['Education'])[0]
    data2['Marital_Status'] = pd.factorize(data['Marital_Status'])[0]

    df = data2.corr().reset_index().rename(columns = {'index': 'Variable 1'})
    df = df.melt('Variable 1', var_name = 'Variable 2', value_name = 'Correlation')

    st.markdown('### EDA Heatmap:')

    base_chart = alt.Chart(df).encode(
        x = 'Variable 1',
        y = 'Variable 2'
    ).properties(
        title = 'Correlation Matrix between Different Features',
        width = 750,
        height = 750
    )

    heatmap = base_chart.mark_rect().encode(
        color = alt.Color('Correlation',
                          scale = alt.Scale(scheme = 'viridis', reverse = True)
        )
    )

    text = base_chart.mark_text(fontSize = 8).encode(
        text = alt.Text('Correlation', format = ',.2f'),
        color = alt.condition(
            alt.datum['Correlation'] > 0.5, 
            alt.value('white'),
            alt.value('black')
        )
    )

    final = (heatmap + text).configure_title(
        fontSize = 20,
        offset = 10,
        anchor = 'middle'
    ).configure_axis(
        labelFontSize = 12
    )

    st.altair_chart(final, use_container_width = True, theme = 'streamlit')

    st.markdown('### Customers who accepted promotion in last marketing campaign grouped by categorical variables:')

    category = st.radio('Choose a categorical variable to view the response rate for each subgroup:', ['Education', 'Marital Status'], horizontal = True)
    st.markdown('---')

    if category == 'Marital Status':
        category2 = 'Marital_Status'
    elif category == 'Education':
        category2 = category

    # Dataframe for bar chart
    cat_bool = data.groupby(category2)['Response'].value_counts().reset_index(name = 'Count')

    if category2 == 'Marital_Status':
        cat_bool.rename(columns = {'Marital_Status': 'Marital Status'}, inplace = True)

    cat_bool['Response'].replace({0: 'Did not accept', 1: 'Accepted'}, inplace = True)

    alt_bar_chart = alt.Chart(cat_bool).mark_bar().encode(
        x = 'sum(Count)',
        y = category,
        color = 'Response'
    ).properties(
        title = f'Responses based on {category}',
        width = 700,
        height = 300
    ).configure_title(
        fontSize = 18,
        offset = 30,
        anchor = 'middle'
    ).configure_range(
        category = {'scheme': 'set2'}
    )

    st.altair_chart(alt_bar_chart, use_container_width = True, theme = 'streamlit')

    st.markdown('### EDA Donut Chart - Proportion of customers who accepted promotion in last marketing campaign:')

    st.write('')

    risk_count = data['Response'].value_counts()
    risk_count = risk_count.rename(index = {0: 'Did not accept', 1: 'Accepted'})
    risk_count = risk_count.reset_index().rename(columns = {'index': 'Type', 'Response': 'Count'})

    donut = alt.Chart(risk_count).mark_arc(innerRadius = 80).encode(
        theta = alt.Theta(field = 'Count', type = 'quantitative'),
        color = alt.Color(field = 'Type', type = 'nominal'),
    )

    st.altair_chart(donut, use_container_width = True, theme = 'streamlit')

    st.markdown('---')

def k_means_model(data):
    st.markdown('## 👥 &nbsp; K-Means Clustering Algorithm')

    df = scaled_processing(data).drop('remainder__Response', axis = 1)

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_clusters = st.slider('Select cluster size:', min_value = 2, max_value = 30, value = 3)
        selected_init = st.radio('Select initialisation method of centroids:', ['k-means++', 'random'], horizontal = True)
        selected_n_init = st.slider('Select no. of times algorithm will be run with different centroid seeds:', min_value = 1, max_value = 30, value = 10)
        selected_max_iter = st.slider('Select maximum number of iterations:', min_value = 100, max_value = 500, value = 300)

    # Initialise machine learning model
    kmeans = KMeans(n_clusters = selected_clusters,
                    init = selected_init,
                    n_init = selected_n_init,
                    max_iter = selected_max_iter)
    kmeans.fit(df)

    # Show results
    st.markdown('### 📊 &nbsp; Results')

    true_labels = data['Response']
    cluster_labels = kmeans.labels_

    st.write('')
    
    st.markdown('##### Clustering Report:')
    clustering_report(df, true_labels, cluster_labels)

    st.markdown('---')


def dbscan_model(data):
    st.markdown('## 🖨️ &nbsp; DBSCAN Algorithm')

    df = scaled_processing(data).drop('remainder__Response', axis = 1)

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_eps = st.slider('Select epsilon parameter:', min_value = 0.1, max_value = 2.0, value = 0.5)
        selected_min_samples = st.slider('Select minimum number of samples in a neighborhood:', min_value = 2, max_value = 20, value = 5)

    # Initialise machine learning model
    dbscan = DBSCAN(eps = selected_eps, 
                    min_samples = selected_min_samples)
    dbscan.fit(df)

    # Show results
    st.markdown('### 📊 &nbsp; Results')

    true_labels = data['Response']
    cluster_labels = dbscan.labels_

    st.write('')
    
    st.markdown('##### Clustering Report:')
    clustering_report(df, true_labels, cluster_labels)

    st.markdown('---')


def hierarch_model(data):
    st.markdown('## 🏛️ &nbsp; Hierarchical Clustering Algorithm')

    df = scaled_processing(data).drop('remainder__Response', axis = 1)

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_clusters = st.slider('Select cluster size:', min_value = 2, max_value = 30, value = 3)
        selected_linkage = st.radio('Select linkage criterion to use:', ['ward', 'complete', 'average', 'single'], horizontal = True)

        if selected_linkage != 'ward':
            selected_affinity = st.radio('Select metric to use for pairwise distances between samples:', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'], horizontal = True)
        else:
            selected_affinity = st.radio('Select metric to use for pairwise distances between samples:', ['euclidean'])

    # Initialise machine learning model
    hierarchical_clustering = AgglomerativeClustering(n_clusters = selected_clusters, 
                                                      linkage = selected_linkage, 
                                                      affinity = selected_affinity)
    hierarchical_clustering.fit(df)

    # Show results
    st.markdown('### 📊 &nbsp; Results')

    true_labels = data['Response']
    cluster_labels = hierarchical_clustering.labels_

    st.write('')
    
    st.markdown('##### Clustering Report:')
    clustering_report(df, true_labels, cluster_labels)

    st.markdown('---')


def gaussian_model(data):
    st.markdown('## 🧪 &nbsp; Gaussian Mixture Model (GMM)')

    df = scaled_processing(data).drop('remainder__Response', axis = 1)

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_components = st.slider('Select number of mixture components:', min_value = 2, max_value = 30, value = 3)
        selected_covariance = st.radio('Select type of covariance matrix to use:', ['full', 'tied', 'diag', 'spherical'], horizontal = True)

    # Initialise machine learning model
    gmm = GaussianMixture(n_components = selected_components, 
                          covariance_type = selected_covariance)
    gmm.fit(df)

    # Show results
    st.markdown('### 📊 &nbsp; Results')

    true_labels = data['Response']
    cluster_labels = gmm.predict(df)

    st.write('')
    
    st.markdown('##### Clustering Report:')
    clustering_report(df, true_labels, cluster_labels)

    st.markdown('---')


def spectral_model(data):
    st.markdown('## 👻 &nbsp; Spectral Clustering Algorithm')

    df = scaled_processing(data).drop('remainder__Response', axis = 1)

    st.write('')

    with st.sidebar:
        st.markdown('# 🔢 &nbsp; User Inputs')
        selected_clusters = st.slider('Select cluster size:', min_value = 2, max_value = 30, value = 3)
        selected_affinity = st.radio('Select affinity matrix to use:', ['rbf', 'nearest_neighbors'], horizontal = True)

    # Initialise machine learning model
    spectral_clustering = SpectralClustering(n_clusters = selected_clusters, 
                                             affinity = selected_affinity)
    spectral_clustering.fit(df)

    # Show results
    st.markdown('### 📊 &nbsp; Results')

    true_labels = data['Response']
    cluster_labels = spectral_clustering.labels_

    st.write('')
    
    st.markdown('##### Clustering Report:')
    clustering_report(df, true_labels, cluster_labels)

    st.markdown('---')


if __name__ == "__main__":
    st.set_page_config(page_title = 'Customer Segment ML App', page_icon = '🛒')
    main()