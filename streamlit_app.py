import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

def apply_anomaly_detection_KMeans(data, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    distances = kmeans.transform(data).min(axis=1)
    threshold = distances.mean() + 3 * distances.std()
    anomalies = distances > threshold
    data_with_anomalies = data.copy()
    data_with_anomalies['Anomaly_KMeans'] = anomalies.astype(int)
    return data_with_anomalies

def elbow_method(data, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 10):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Streamlit application code
if __name__ == "__main__":
    st.title("Anomaly Detection with Machine Learning Algorithms")

    selected_anomalyAlgorithm = st.selectbox("Select Anomaly Detection Algorithm:", ["Isolation Forest", "K Means Clustering"])

    if selected_anomalyAlgorithm == "K Means Clustering":
        st.markdown(
            "<h2 style='font-size: 24px; color: blue;'>Upload Dataset, Empower Machine Learning Algorithms!</h2>",
            unsafe_allow_html=True)
        data_file = st.file_uploader("Upload File", type=["csv", "xlsx", "XLSX"])

        if data_file is not None:
            file_extension = data_file.name.split(".")[-1]
            if file_extension == "csv":
                data = pd.read_csv(data_file, encoding='ISO-8859-1')
            elif file_extension in ["xlsx", "XLSX"]:
                data = pd.read_excel(data_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")

            copy_data = data.copy()
            st.write("Dealing with missing values:")
            threshold = 0.1
            missing_percentages = data.isnull().mean()
            columns_to_drop = missing_percentages[missing_percentages > threshold].index
            data = data.drop(columns=columns_to_drop)
            st.write(f"Features with more than {threshold*100:.2f}% missing values dropped successfully.")

            st.write("Dealing with duplicate values...")
            num_duplicates = data.duplicated().sum()
            data_unique = data.drop_duplicates()
            st.write(f"Number of duplicate rows: {num_duplicates}")
            st.write("Dealing done with duplicates.")

            st.write("Performing categorical feature encoding...")
            categorical_features = [feature for feature in data_unique.columns if data_unique[feature].dtype == 'object']
            data_encoded = data_unique.copy()
            for feature in categorical_features:
                labels_ordered = data_unique.groupby([feature]).size().sort_values().index
                labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
                data_encoded[feature] = data_encoded[feature].map(labels_ordered)
            data = data_encoded
            st.write("Categorical features encoded successfully.")

            st.write("Performing feature scaling...")
            numeric_columns = data.select_dtypes(include=["int", "float"]).columns

            if len(numeric_columns) == 0:
                st.write("No numeric columns found.")
            else:
                scaler = MinMaxScaler()
                data_scaled = data.copy()
                data_scaled[numeric_columns] = scaler.fit_transform(data_scaled[numeric_columns])
                data = data_scaled
                st.write("Feature scaling performed successfully.")

            st.write("Downloading the dataset...")

            modified_dataset_filename = "modified_dataset.csv"
            st.write(data.head())
            st.write(data.shape)

            # Hyperparameter tuning
            st.subheader("Hyperparameter Tuning")
            max_clusters = st.slider("Select max number of clusters for Elbow Method", 1, 20, 10)
            if st.button("Run Elbow Method"):
                wcss = elbow_method(data, max_clusters=max_clusters)
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, max_clusters + 1), wcss, marker='o')
                plt.xlabel("Number of Clusters")
                plt.ylabel("WCSS")
                plt.title("Elbow Method")
                st.pyplot(plt)

            n_clusters = st.slider("Select number of clusters for K-means", 1, 20, 8)
            data_with_anomalies_KMeans = apply_anomaly_detection_KMeans(data, n_clusters=n_clusters)

            data_with_anomalies_KMeans['PointColor'] = 'Inlier'
            data_with_anomalies_KMeans.loc[data_with_anomalies_KMeans['Anomaly_KMeans'] == 1, 'PointColor'] = 'Outlier'

            AnomalyFeature = data_with_anomalies_KMeans[["Anomaly_KMeans"]]

            st.subheader("Data with Anomalies")
            final_data = pd.concat([copy_data, AnomalyFeature], axis=1)
            st.write(final_data.head(5))

            st.subheader("Visualize anomalies")
            selected_option = st.radio("Please select the type of plot:", ["2D ScatterPlot", "3D ScatterPlot", "Density Plot", "Parallel Coordinates Plot", "QQ-Plot"])

            if selected_option == "QQ-Plot":
                selected_x_col = st.selectbox("Select X-axis column", data.columns)
                selected_data = data_with_anomalies_KMeans[selected_x_col]
                sm.qqplot(selected_data, line='s')
                plt.xlabel('Theoretical Quantiles')
                plt.ylabel(f'Quantiles of {selected_x_col}')
                plt.title(f'QQ-Plot of {selected_x_col}')
                plt.gca().set_facecolor('#F1F6F5')
                st.pyplot(plt)

            elif selected_option == "Density Plot":
                selected_x_col = st.selectbox("Select X-axis column", data.columns)
                sns.kdeplot(data_with_anomalies_KMeans[selected_x_col], shade=True)
                plt.xlabel(f'{selected_x_col} Label')
                plt.ylabel('Density')
                plt.title(f'Density Plot of {selected_x_col}')
                plt.gca().set_facecolor('#F1F6F5')
                st.pyplot(plt)

            elif selected_option == "Parallel Coordinates Plot":
                selected_columns = st.multiselect("Select columns for Parallel Coordinates Plot", data.columns)
                if len(selected_columns) > 0:
                    parallel_data = final_data[selected_columns + ["Anomaly_KMeans"]]
                    fig = px.parallel_coordinates(
                        parallel_data,
                        color="Anomaly_KMeans",
                        color_continuous_scale=["blue", "red"],
                        labels={"Anomaly_KMeans": "Anomaly"},
                    )
                    fig.update_layout(
                        title="Parallel Coordinates Plot",
                        paper_bgcolor='#F1F6F5',
                        plot_bgcolor='white',
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select at least one column for the Parallel Coordinates Plot.")

            elif selected_option == "2D ScatterPlot":
                selected_x_col = st.selectbox("Select X-axis column", data.columns)
                selected_y_col = st.selectbox("Select Y-axis column", data.columns)
                fig = px.scatter(
                    data_with_anomalies_KMeans,
                    x=selected_x_col,
                    y=selected_y_col,
                    color="PointColor",
                    color_discrete_map={"Inlier": "blue", "Outlier": "red"},
                    title='K-Means Clustering Anomaly Detection',
                    labels={selected_x_col: selected_x_col, "Anomaly_KMeans": 'Anomaly_KMeans', "PointColor": "Data Type"},
                )
                fig.update_traces(
                    marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
                    selector=dict(mode='markers+text')
                )
                fig.update_layout(
                    legend=dict(
                        itemsizing='constant',
                        title_text='',
                        font=dict(family='Arial', size=12),
                        borderwidth=2
                    ),
                    xaxis=dict(
                        title_text=selected_x_col,
                        title_font=dict(size=14),
                        showgrid=False,
                        showline=True,
                        linecolor='lightgray',
                        linewidth=2,
                        mirror=True
                    ),
                    yaxis=dict(
                        title_text=selected_y_col,
                        title_font=dict(size=14),
                        showgrid=False,
                        showline=True,
                        linecolor='lightgray',
                        linewidth=2,
                        mirror=True
                    ),
                    title_font=dict(size=18, family='Arial'),
                    paper_bgcolor='#F1F6F5',
                    plot_bgcolor='white',
                    margin=dict(l=80, r=80, t=50, b=80),
                )
                st.plotly_chart(fig)

                import time
                with st.spinner('Wait for it...'):
                    time.sleep(3)
                st.success('Done!')

                st.download_button(
                    label="Download Plot (HTML)",
                    data=fig.to_html(),
                    file_name='plot.html',
                    mime='text/html'
                )

            elif selected_option == "3D ScatterPlot":
                selected_x_col = st.selectbox("Select X-axis column", data.columns)
                selected_y_col = st.selectbox("Select Y-axis column", data.columns)
                selected_z_col = st.selectbox("Select Z-axis column", data.columns)
                fig = px.scatter_3d(
                    data_with_anomalies_KMeans,
                    x=selected_x_col,
                    y=selected_y_col,
                    z=selected_z_col,
                    color="PointColor",
                    color_discrete_map={"Inlier": "blue", "Outlier": "red"},
                    title='K-Means Clustering Anomaly Detection (3D Scatter Plot)',
                    labels={selected_x_col: selected_x_col, selected_y_col: selected_y_col, selected_z_col: selected_z_col, "Anomaly_KMeans": 'Anomaly_KMeans', "PointColor": "Data Type"},
                )
                fig.update_traces(
                    marker=dict(size=8, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')),
                    selector=dict(mode='markers+text')
                )
                fig.update_layout(
                    legend=dict(
                        itemsizing='constant',
                        title_text='',
                        font=dict(family='Arial', size=12),
                        borderwidth=2
                    ),
                    scene=dict(
                        xaxis=dict(
                            title_text=selected_x_col,
                            title_font=dict(size=14),
                        ),
                        yaxis=dict(
                            title_text=selected_y_col,
                            title_font=dict(size=14),
                        ),
                        zaxis=dict(
                            title_text=selected_z_col,
                            title_font=dict(size=14),
                        ),
                    ),
                    title_font=dict(size=18, family='Arial'),
                    paper_bgcolor='#F1F6F5',
                    plot_bgcolor='white',
                    margin=dict(l=80, r=80, t=50, b=80),
                )
                st.plotly_chart(fig)

            import time
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success('Done!')

            st.write("Download the data with anomaly indicator")
            st.download_button(
                label="Download",
                data=final_data.to_csv(index=False),
                file_name="KMeansAnomaly.csv",
                mime="text/csv"
            )

            filtered_data = final_data[final_data['Anomaly_KMeans'] == 1]
            st.write("Download the dataset where all observations are labeled as anomalies")
            st.download_button(
                label="Download",
                data=filtered_data.to_csv(index=False),
                file_name="KMeansOnlyAnomaly.csv",
                mime="text/csv"
            )

            num_anomalies = data_with_anomalies_KMeans['Anomaly_KMeans'].sum()
            total_data_points = len(data_with_anomalies_KMeans)
            percentage_anomalies = (num_anomalies / total_data_points) * 100

            st.write(f"Number of anomalies: {num_anomalies}")
            st.write(f"Percentage of anomalies: {percentage_anomalies:.2f}%")
