import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import squarify
import plotly.express as px

# ----- Cài đặt trang
st.set_page_config(page_title="Phân khúc khách hàng", page_icon=":👥:", layout="centered")
st.image("images/Customer segment.jpg")
st.title("Dự án Khoa học Dữ liệu")
st.header("Phân khúc khách hàng trong Bán lẻ trực tuyến")
st.write("""
         #### 👩🏻‍🏫 Giáo viên hướng dẫn: Khuất Thùy Phương ####
         #### 🏫 Nhóm: Huỳnh Văn Tài - Trần Thế Lâm ####
         """)
home = """
### Phân cụm khách hàng là gì?
* Phân cụm khách hàng (customer segmentation) là quá trình phân chia khách hàng dựa trên các đặc điểm chung như hành vi, thói quen mua sắm và sử dụng dịch vụ của họ,... để các công ty, doanh nghiệp có thể tiếp thị cho từng nhóm khách hàng một cách hiệu quả và phù hợp hơn.
### Tại sao phải phân cụm khách hàng?
* Bởi vì bạn không thể đối xử với họ giống như nhau, sử dụng cùng một loại nội dung, cùng một kênh truyền thông và cùng độ ưu tiên. 
* Phân cụm giúp hiểu rõ hơn về từng nhóm khách hàng về nhu cầu, sở thích, hành vi mua sắm,... từ đó tiếp thị hiệu quả hơn, tăng tỷ lệ chuyển đổi, tối ưu hóa chi phí marketing.
### Phân chia cụm bằng cách nào?
* Có thể dựa trên độ tuổi, mức thu nhập, số lượng con cái, mức chi tiêu hằng tháng, số lượng đơn hàng đã mua, vùng miền, thiết bị sử dụng (Android hay iOS),...
* Mỗi khách hàng có hành vi và đặc điểm (attribute) khác nhau nhưng nhìn chung vẫn có mối liên hệ mục đích của chúng ta là tìm ra các mối liên hệ đó và nhóm họ lại.
### Phân cụm khách hàng theo RFM: 
* Recency (R): Số ngày từ lần mua hàng cuối cùng đến hiện tại. Khách hàng càng mua hàng gần đây thì càng dễ gắn kết với thương hiệu hơn so với những người lâu rồi không quay lại mua.
* Frequency (F): Tần suất mua hàng, tức là tổng số đơn hàng của khách hàng.
* Monetary (M): Tổng số tiền khách hàng đã chi tiêu để mua hàng, tức là tổng giá trị của các đơn hàng của khách hàng.
* Mô hình RFM sử dụng 3 yếu tố chính này để phân loại khách hàng thành các nhóm. 
* Tại dự án 3, chúng tôi áp dụng trên bộ Retail Online (Link tải đầu trang).

### 🏗️ **Cách Thực hiện**

* Tiền xử lý dữ liệu (ví dụ: làm sạch dữ liệu, xử lý dữ liệu trùng, NA, outlier, tạo ra các feauture mới, scaler...).
* Tính toán RFM (Recency, Frequency, Monetary).
* Sử dụng KMeans để phân cụm khách hàng.

### 🎯 **Tính Năng Chính**

* Dự đoán phân khúc khách hàng cho dữ liệu RFM được tải lên.
* Nhập các RFM cụ thể để dự đoán phân khúc.
* Dự đoán phân khúc khách hàng cho dữ liệu ID Customer được tải lên.
* Nhập các ID khách hàng cụ thể để dự đoán phân khúc.
* Phân cụm khách hàng cho tập dữ liệu mới.

### 🚀 **Bắt Đầu**
1. Tải lên dữ liệu khách hàng, sales của bạn (định dạng CSV) hoặc nhập các ID, RFM khách hàng.
2. Nhấp vào "Dự đoán" để dự đoán phân khúc cho mỗi khách hàng.
"""
@st.cache_data
def load_data():
    return pd.read_csv("data/OnlineRetail.csv", encoding='latin-1')
data = load_data()

# Chuyển DataFrame thành byte
csv_file = data.to_csv(index=False, encoding='utf-8')
csv_file = csv_file.encode('utf-8')  # Chuyển đổi thành byte

if st.download_button(label="Tải xuống bộ dữ liệu Retail Online", data=csv_file, file_name="OnlineRetail.csv", help="Nhấp để tải xuống tập dữ liệu gốc của Retail Online"):
    st.text("Tải Dữ liệu Thành công")

# Chạy mô hình
with open('./models/kmean_model.pkl', 'rb') as file:
    model_kmeans_lds6 = pickle.load(file)
scaler = joblib.load('models/scaler.pkl')

# Loại bỏ ngoại lệ
import pandas as pd
def remove_outliers_iqr(df, X):
    for col in X:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outliers_ = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if outliers_.shape[0] > 0:
            df.drop(outliers_.index, axis=0, inplace=True)

# Chỉnh sửa tập dữ liệu
def get_dataset_clean(data1):
    # Kiểm tra xem data1 có phải là DataFrame không
    if not isinstance(data1, pd.DataFrame):
        raise ValueError("Đầu vào không phải là DataFrame")
    # Loại bỏ các dòng có CustomerID thiếu
    data1.dropna(subset=['CustomerID'], inplace=True)
    # Loại bỏ các dòng có InvoiceNo bắt đầu bằng 'C'
    data1 = data1.loc[~data1["InvoiceNo"].str.startswith('C', na=False)]
    data1 = data1.drop(data1[data1['UnitPrice'] < 0].index)  
    data1 = data1.drop(data1[data1['Quantity'] < 0].index)
    input_col_num= ['Quantity', 'UnitPrice']
    remove_outliers_iqr(data1, input_col_num)
    # Lấy mode của UnitPrice cho mỗi StockCode
    unitprice_mode = data1.groupby('StockCode')['UnitPrice'].apply(lambda x: x.mode().iloc[0])   
    # Thay thế UnitPrice bằng mode nếu bằng 0 hoặc null
    data1['UnitPrice'] = data1.apply(lambda row: unitprice_mode[row['StockCode']] if row['UnitPrice'] == 0 or pd.isnull(row['UnitPrice']) else row['UnitPrice'], axis=1)
    # Chuyển InvoiceDate thành datetime và trích xuất các feauture khác
    data1['X'] = pd.to_datetime(data1['InvoiceDate'], format='%d-%m-%Y %H:%M')
    data1['InvoiceDate'] = pd.to_datetime(data1['InvoiceDate'], format='%d-%m-%Y %H:%M').dt.date
    data1['InvoiceDate'] = data1['X'].dt.date
    data1['Year'] = data1['X'].dt.year
    data1['Month'] = data1['X'].dt.month
    data1['Day'] = data1['X'].dt.day_name()
    data1['Hour'] = data1['X'].dt.hour
    data1 = data1.drop(columns=['X'])
    # Tính toán Doanh thu
    data1['TotalPrice'] = data1['Quantity'] * data1['UnitPrice']
    return data1

# Vẽ biểu đồ theo doanh thu
def plot_totalprice_chart(chart_type):
    if chart_type == 'Theo Giờ':
        groupby_col = 'Hour'
        xlabel = 'Giờ'
    elif chart_type == 'Theo Ngày':
        groupby_col = 'Day'
        xlabel = 'Ngày'
    elif chart_type == 'Theo Tháng':
        groupby_col = 'Month'
        xlabel = 'Tháng'
    elif chart_type == 'Theo Năm':
        groupby_col = 'Year'
        xlabel = 'Năm'
    elif chart_type == 'Theo Quốc Gia':
        groupby_col = 'Country'
        xlabel = 'Quốc Gia'
    # Tạo biểu đồ
    plt.figure(figsize=(12, 6))
    df.groupby(groupby_col)['TotalPrice'].sum().plot(kind='bar')
    plt.title(f'Doanh Thu {chart_type}')
    plt.xlabel(xlabel)
    plt.ylabel('Doanh Thu')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt

def predict_segmentKmean_data(data1):
    # Tính toán giá trị RFM
    max_date = data1['InvoiceDate'].max()
    recency = data1.groupby('CustomerID')['InvoiceDate'].max().apply(lambda x: (max_date - x).days)
    frequency = data1.groupby('CustomerID')['InvoiceNo'].nunique()
    monetary = data1.groupby('CustomerID')['TotalPrice'].sum()
    # Tạo DataFrame RFM
    rfm_values = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
    # Scale RFM values
    rfm_values_scaled = scaler.transform(rfm_values)
    # Tải mô hình và dự đoán cụm
    clusters = model_kmeans_lds6.predict(rfm_values_scaled)
    # Ánh xạ cụm thành phân khúc        
    segments = {0: 'Lost', 1: 'Big spender', 2: 'At risk', 3: 'Regular'}
    segment_name = [segments[i] for i in clusters]
    rfm_values['Segment'] = segment_name
    return segment_name, rfm_values
    
# Ứng dụng 1. Nhập CustomerID để xác định segment của khách hàng
def predict_segmentKmean(CustomerID, data):
    if CustomerID not in data['CustomerID'].values:
        return 'Không tìm thấy khách hàng'
    else: 
        max_date = data['InvoiceDate'].max()
        customer_data = data[data['CustomerID'] == CustomerID]
        Recency = (max_date - customer_data['InvoiceDate'].max()).days
        Frequency = customer_data['InvoiceNo'].nunique()
        Monetary = customer_data['TotalPrice'].sum()
        # Create the RFM DataFrame with columns in the exact order as during the scaler's fit
        rfm_values = pd.DataFrame([[Recency, Frequency, Monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        # Load the scaler and scale the RFM values
        scaler = joblib.load('models/scaler.pkl')
        rfm_values = scaler.transform(rfm_values)
        # Load the model and predict the cluster
        model = pickle.load(open('models/kmean_model.pkl', 'rb'))
        cluster = model.predict(rfm_values)
        # Map the cluster to the segment name
        segments = {0: 'Lost', 1: 'Big spender', 2: 'At risk', 3: 'Regular'}
        segment_name = segments.get(cluster[0], 'Unknown segment')
        return segment_name
# Ứng dụng 2. Nhập RFM của khách hàng để dự đoán segment
def predict_segmentKmean2(Recency, Frequency, Monetary):
    # Create the RFM DataFrame with columns in the exact order as during the scaler's fit
    rfm_values = pd.DataFrame([[Recency, Frequency, Monetary]], columns=['Recency', 'Frequency', 'Monetary'])
    # Load the scaler and scale the RFM values
    scaler = joblib.load('models/scaler.pkl')
    rfm_values = scaler.transform(rfm_values)
    # Load the model and predict the cluster
    model = pickle.load(open('models/kmean_model.pkl', 'rb'))
    cluster = model.predict(rfm_values)
    # Map the cluster to the segment name
    segments = {0: 'Lost', 1: 'Big spender', 2: 'At risk', 3: 'Regular'}
    segment_name = segments.get(cluster[0], 'Unknown segment')
    return segment_name

# Vẽ Tree map
def visualize_rfm_squarify(rfm_values):
    rfm_agg = rfm_values.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(0)
    rfm_agg.columns = rfm_agg.columns.droplevel()
    rfm_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
    rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)
    # Reset the index
    rfm_agg = rfm_agg.reset_index()
    # Change the Cluster Columns Datatype into discrete values
    rfm_agg['Segment'] = 'Nhóm ' + rfm_agg['Segment'].astype(str)
    colors_dict_cluster = {'Nhóm Lost':'yellow','Nhóm Big spender':'royalblue', 'Nhóm At risk':'cyan',
               'Nhóm Regular':'red'}
    # Tạo biểu đồ
    fig1, ax = plt.subplots(figsize=(14, 10))
    squarify.plot(sizes= rfm_agg['Count'],
              text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
              color=colors_dict_cluster.values(),
              label=['{} \n{:.0f} ngày \n{:.0f} đơn hàng \n{:.0f} $ \n{:.0f} khách hàng ({}%)'.format(*rfm_agg.iloc[i])
              for i in range(0, len(rfm_agg))], alpha=0.5 )
    plt.title("Phân khúc khách hàng Kmeans", fontsize=26, fontweight="bold")
    plt.axis('off')
    fig2 = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Segment",
           hover_name="Segment", size_max=70, color_discrete_map=colors_dict_cluster)
    # Hiển thị biểu đồ bằng Streamlit
    st.pyplot(fig1)
    st.plotly_chart(fig2)

# GUI. Giao diện người dùng
menu = ["🏠Trang chủ", "👨‍🔬Insign dữ liệu Retail Online", "🛒Dự đoán cho dữ liệu RFM mới", "👨‍💼Dự đoán cho ID khách hàng", "📈RFM cho bộ dữ liệu mới"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == "🏠Trang chủ":    
    st.markdown(home, unsafe_allow_html=True)

if choice == "👨‍🔬Insign dữ liệu Retail Online":    
    df1 = data.copy()
    df = get_dataset_clean(df1)
    st.write("""
    ### **Giải Thích** ###
    - InvoiceNo: Mã số hóa đơn.
    - StockCode: Mã số sản phẩm.
    - Description: Mô tả sản phẩm.
    - Quantity: Số lượng hàng hóa mua trên hóa đơn.
    - UnitPrice: Giá sản phẩm.
    - InvoiceDate: Thời gian giao dịch được thực hiện.
    - CustomerID: Mã định danh khách hàng.
    - Country: Quốc gia.
    """)
    st.subheader("Dữ liệu")
    st.write(df.head())
    # Hiển thị thống kê cơ bản
    st.subheader("Thống Kê Cơ Bản")
    st.write(df.describe())
    # Giao diện người dùng Streamlit
    st.subheader('Biểu đồ Doanh thu')
    # Dropdown để chọn loại biểu đồ
    chart_type = st.selectbox('Chọn loại biểu đồ:', ['Theo Giờ', 'Theo Ngày', 'Theo Tháng', 'Theo Năm', 'Theo Quốc Gia'])
    # Hiển thị biểu đồ doanh thu dựa trên loại biểu đồ đã chọn
    plot_totalprice_chart(chart_type)
    plt = plot_totalprice_chart(chart_type)
    st.pyplot(plt)
    st.write("""
    ### Nhận xét ###
    - Doanh thu cao nhất vào 12 trưa trong ngày.
    - Không có doanh thu vào thứ 7, có thể do đó là ngày nghỉ của công ty.
    - Doanh thu tăng vào các tháng cuối năm, có thể do mùa lễ hội.
    - Doanh thu từ UK chiếm tỷ lệ cao nhất, chiếm 90% tổng doanh thu.      
    """)

# Giải thích RFM và phân nhóm khách hàng
elif choice == "🛒Dự đoán cho dữ liệu RFM mới":
    st.write("""
    ### RFM
    Số lượng nhóm khách hàng có thể thay đổi tùy thuộc vào cách định nghĩa của bạn. Ở mức đơn giản nhất chúng ta sẽ chia thành 4 nhóm dưới đây:
    1. **Nhóm Lost (Lost):** Bao gồm các khách hàng đã lâu không quay lại mua hàng, số lượng đơn hàng ít và tổng giá trị đơn hàng thấp.
    2. **Nhóm Big Spender (Big Spender):** Là nhóm khách hàng có mua hàng gần đây, số lượng đơn hàng nhiều và tổng giá trị đơn hàng cao.
    3. **Nhóm At Risk (At Risk):** Bao gồm các khách hàng mua hàng không thường xuyên, có nguy cơ chuyển sang nhóm "Lost" nếu không có các hoạt động kích thích mua hàng.
    4. **Nhóm Regular (Regular):** Là nhóm khách hàng với tần suất và giá trị mua hàng trung bình, không quá cao cũng không quá thấp.
    """)
    # Tiêu đề cho phần nhập dữ liệu RFM
    st.subheader("Lựa chọn dữ liệu")
    type = st.radio("Tải lên tệp RFM hoặc nhập RFM?", options=("Tải lên tệp RFM", "Nhập RFM"))   
    if type == "Tải lên tệp RFM":
        st.write("Vui lòng tải lên tệp CSV chứa dữ liệu RFM của khách hàng. Đảm bảo rằng tệp đã tải lên có định dạng đúng.")
        # Hiển thị bảng để hướng dẫn khách hàng nhập dữ liệu RFM
        example_rfm_table = pd.DataFrame({
            "Recency": ["Nhập số ngày kể từ lần mua hàng gần nhất", "Nhập số ngày kể từ lần mua hàng gần nhất"],
            "Frequency": ["Nhập số lượng đơn hàng", "Nhập số lượng đơn hàng"],
            "Monetary": ["Nhập tổng giá trị tiền đã chi tiêu", "Nhập tổng giá trị tiền đã chi tiêu"]
        })
        st.table(example_rfm_table)
        uploaded_file = st.file_uploader("Chọn tệp", type=['csv'])
        if uploaded_file is not None:
            st.write(f"**Kết quả phân cụm:**")
            data_rfm = pd.read_csv(uploaded_file, encoding='latin-1')
            if st.button("Dự đoán"):
                try:
                    # Tiêu chuẩn hóa dữ liệu RFM mới
                    data_rfm_sca = scaler.transform(data_rfm)
                    # Dự đoán nhóm của khách hàng mới
                    cluster = model_kmeans_lds6.predict(data_rfm_sca)
                    # Tạo dictionary ánh xạ cluster sang segment
                    segments = {0: 'Lost', 1: 'Big spender', 2: 'At risk', 3: 'Regular'}
                    # Tạo cột "Cluster" mới trong DataFrame data_rfm
                    data_rfm['Segment'] = [segments.get(segment_index, 'Không xác định') for segment_index in cluster]
                    data_rfm
                    # Tạo nút để tải DataFrame về dưới dạng CSV
                    csv_file = data_rfm.to_csv(index=False, encoding='utf-8')
                    csv_file = csv_file.encode('utf-8')  # Chuyển đổi thành bytes
                    if st.download_button(label="Tải về các dự đoán đã nhập", data=csv_file, file_name="RFM_predictions.csv", help="Nhấp để tải về dữ liệu dự đoán dưới dạng CSV"):
                        st.text("Dữ liệu đã được tải về thành công.")
                except ValueError:
                    st.error("Dữ liệu nhập không chính xác.")
            pass

    # Tạo DataFrame để lưu giá trị RFM của khách hàng mới
    if type == "Nhập RFM":
        data_rfm = pd.DataFrame(columns=["Recency", "Frequency", "Monetary"])
        # Dùng vòng lặp để tạo slider cho từng khách hàng
        for i in range(2):
            st.write(f"Khách hàng {i+1}")
            # Tạo các slider để nhập giá trị cho cột Recency, Frequency, Monetary
            recency = st.slider("Recency", 1, 365, 100, key=f"recency_{i}")
            frequency = st.slider("Frequency", 1, 50, 5, key=f"frequency_{i}")
            monetary = st.slider("Monetary", 1, 10000, 100, key=f"monetary_{i}")
            new_customer_data = {"Recency": recency, "Frequency": frequency, "Monetary": monetary}
            # Thêm dữ liệu của khách hàng mới vào DataFrame
            data_rfm = pd.concat([data_rfm, pd.DataFrame(new_customer_data, index=[0])], ignore_index=True)
        # Button để thực hiện dự đoán
        if st.button("Dự đoán"):
            try:
                for i, row in data_rfm.iterrows():
                    segment = predict_segmentKmean2(row["Recency"], row["Frequency"], row["Monetary"])
                    st.write(f"**Phân khúc dự đoán cho Khách hàng {i+1}**")
                    segment
            except ValueError:
                st.error("Dữ liệu nhập không chính xác.")
        pass

elif choice ==  "👨‍💼Dự đoán cho ID khách hàng":
    st.subheader("Lựa chọn dữ liệu")
    type = st.radio("Tải lên tệp danh sách hoặc nhập ID khách hàng?", options=("Tải lên tệp danh sách ID khác hàng", "Nhập ID khác hàng"))   
    if type == "Tải lên tệp danh sách ID khác hàng":
        uploaded_file_1 = st.file_uploader("Chọn tệp", type=['csv'])
        if uploaded_file_1 is not None:
            df1 = pd.read_csv(uploaded_file_1, encoding='latin-1')
            df = get_dataset_clean(df1)
            st.write('Dữ liệu đã nhập')
            st.dataframe(df)
            lines = df.iloc[:, df.columns.get_loc('CustomerID')].astype(int)
            lines = np.array(lines)
            predictions = []
            for ids in lines:
                segment = predict_segmentKmean(ids, df)
                predictions.append((ids, segment))
            df['Segment'] = [pred[1] for pred in predictions]  # Adding 'segment' column to the DataFrame
            st.write("Phân khúc dự đoán cho Khách hàng")
            df
            # Tạo nút để tải DataFrame về dưới dạng CSV
            csv_file = df.to_csv(index=False, encoding='utf-8')
            csv_file = csv_file.encode('utf-8')  # Chuyển đổi thành bytes
            if st.download_button(label="Tải về các dự đoán đã nhập", data=csv_file, file_name="ID_predictions.csv", help="Nhấp để tải về dữ liệu dự đoán dưới dạng CSV"):
                st.text("Dữ liệu đã được tải về thành công.")
    if type == "Nhập ID khác hàng":
        st.write("""
            #### Hướng dẫn:
            1. Nhập ID khách hàng (CustomerID) vào ô tìm kiếm, mỗi ID cách nhau bằng dấu phẩy.
            2. Nhấn nút "Dự đoán" để xem kết quả.
            #### Ví dụ:
            - Bước 1: Nhập 15780, 12468, 28769 vào ô tìm kiếm
            - Bước 2: Nhấn nhút Dự đoán
        """)
        CustomerID = st.text_area(label="Nhập CustomerID (phân tách bằng dấu phẩy):")
        predict_button = st.button("Dự đoán")
        df = pd.read_csv('data/OnlineRetail.csv', encoding='ISO-8859-1')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M').dt.date
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        if predict_button:
            if CustomerID:
                try:
                    CustomerID = [int(id.strip()) for id in CustomerID.split(",")]
                    lines = np.array(CustomerID)
                    for CustomerID in lines:
                        segment = predict_segmentKmean(CustomerID, df)
                        st.code(f"Dự đoán phân cụm ID Customer {CustomerID}: ")
                        segment
                except ValueError:
                    st.error("Dữ liệu không hợp lê, vui lòng nhập lại.")

elif choice == "📈RFM cho bộ dữ liệu mới":
    st.subheader("Upload dữ liệu: ")
    st.write('Dữ liệu upload phải có cấu trúc tương tự Retail Online')
    df = data.copy()
    st.write(df.head())
    uploaded_file = st.file_uploader("Chọn tập tin", type=['csv'])
    if uploaded_file is not None:
        st.write(f"**Phân khúc được Dự đoán cho Dữ liệu mới:**")
        df = pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        st.write(f"**Phân khúc được Dự đoán cho Retail Data:**")
    df1 = get_dataset_clean(df)
    segment_name, rfm_values = predict_segmentKmean_data(df1)
    st.write(f"**Tóm tắt Phân khúc Dự đoán:**")
    st.write(rfm_values['Segment'].value_counts())
    # Trực quan hóa dữ liệu
    st.subheader("Trực Quan Hóa Dữ liệu")
    # Biểu đồ Treemap cho phân khúc khách hàng
    st.write("Biểu Đồ Treemap Phân Khúc Khách Hàng")
    visualize_rfm_squarify(rfm_values)
