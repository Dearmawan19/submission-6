# Laporan Proyek Machine Learning - Dearmawan

## Domain Proyek

Diabetes merupakan penyakit kronis yang ditandai dengan tingginya kadar gula darah (glukosa). Penyakit ini menjadi masalah kesehatan global yang signifikan karena prevalensinya yang terus meningkat dan komplikasinya yang serius, seperti penyakit jantung, stroke, gagal ginjal, kebutaan, dan amputasi. Menurut World Health Organization (WHO), diabetes adalah penyebab utama kebutaan, gagal ginjal, serangan jantung, stroke, dan amputasi tungkai bawah [1]. Deteksi dini dan pengelolaan diabetes sangat penting untuk mencegah atau menunda komplikasi tersebut.

**Masalah:** Banyak kasus diabetes, terutama tipe 2, tidak terdiagnosis hingga komplikasi muncul. Keterlambatan diagnosis ini meningkatkan beban penyakit baik bagi individu maupun sistem kesehatan. Oleh karena itu, diperlukan suatu cara untuk membantu mengidentifikasi individu yang berisiko tinggi terkena diabetes berdasarkan data klinis rutin yang lebih mudah diakses.

**Pentingnya Solusi:** Dengan memanfaatkan teknik machine learning, kita dapat membangun model prediktif yang menganalisis data pasien (seperti kadar glukosa, tekanan darah, indeks massa tubuh, usia, dll.) untuk memperkirakan kemungkinan seseorang menderita diabetes. Model ini dapat berfungsi sebagai alat bantu skrining awal, membantu tenaga medis untuk memprioritaskan pasien yang memerlukan tes diagnostik lebih lanjut (seperti tes HbA1c atau tes toleransi glukosa oral), sehingga memungkinkan intervensi dini dan pengelolaan penyakit yang lebih efektif.

**Referensi:**
[1] World Health Organization. (2023). *Diabetes*. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/diabetes

## Business Understanding

Bagian ini menjelaskan proses klarifikasi masalah yang ingin diselesaikan.

### Problem Statements

Berdasarkan latar belakang di atas, masalah yang ingin diselesaikan adalah:

1.  Bagaimana cara membangun model machine learning yang efektif untuk memprediksi keberadaan penyakit diabetes pada pasien berdasarkan fitur-fitur diagnostik yang tersedia dalam dataset?
2.  Algoritma klasifikasi manakah (misalnya antara K-Nearest Neighbors, Logistic Regression, dan Random Forest) yang memberikan performa terbaik dalam kasus prediksi diabetes ini, berdasarkan metrik evaluasi yang relevan?

### Goals

Tujuan dari proyek ini adalah:

1.  Mengembangkan model klasifikasi machine learning yang mampu memprediksi apakah seorang pasien menderita diabetes (Outcome = 1) atau tidak (Outcome = 0) dengan tingkat akurasi dan recall yang baik.
2.  Membandingkan performa beberapa algoritma klasifikasi (KNN, Random Forest, dan Logistic Regression) untuk menentukan algoritma yang paling optimal berdasarkan metrik evaluasi seperti Accuracy, Precision, Recall, dan F1-Score.

### Solution Statements

Untuk mencapai tujuan tersebut, solusi yang diajukan adalah:

1.  **Menggunakan Algoritma Klasifikasi:** Menerapkan setidaknya tiga algoritma klasifikasi yang berbeda pada data yang telah diproses. Dalam proyek ini, algoritma yang akan digunakan dan dibandingkan adalah:
    * K-Nearest Neighbors (KNN): Algoritma berbasis jarak yang sederhana dan intuitif.
    * Random Forest: Algoritma ensemble berbasis pohon keputusan yang cenderung robust dan memberikan performa yang baik.
    * Logistic Regression: Algoritma linear yang umum digunakan untuk klasifikasi biner, memberikan probabilitas.
2.  **Evaluasi Berbasis Metrik:** Mengukur kinerja setiap model (baseline dan tuned) pada data uji menggunakan metrik Accuracy, Precision, Recall, dan F1-Score, serta Mean Squared Error (MSE) untuk klasifikasi biner. Menggunakan Confusion Matrix dan Classification Report untuk analisis yang lebih mendalam. Memilih model terbaik berdasarkan metrik yang paling relevan dengan konteks masalah (misalnya, Recall mungkin lebih diprioritaskan untuk meminimalkan kasus false negative dalam diagnosis medis).

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Diabetes Dataset" yang bersumber dari Kaggle. Dataset ini awalnya berasal dari National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) dan bertujuan untuk memprediksi secara diagnostik apakah seorang pasien menderita diabetes berdasarkan pengukuran diagnostik tertentu. Dataset ini khusus mencakup pasien wanita Pima Indian berusia minimal 21 tahun.

**Sumber Data:** [https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

Dataset ini terdiri dari **768 sampel** (baris data) dan **9 fitur** (kolom), termasuk variabel target ('Outcome'). Semua fitur adalah numerik.

### Variabel-variabel pada Dataset:

* **Pregnancies:** Jumlah kehamilan. (Numerik)
* **Glucose:** Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral. (Numerik)
* **BloodPressure:** Tekanan darah diastolik (mm Hg). (Numerik)
* **SkinThickness:** Ketebalan lipatan kulit trisep (mm). (Numerik)
* **Insulin:** Insulin serum 2 jam (mu U/ml). (Numerik)
* **BMI:** Indeks Massa Tubuh (berat dalam kg / (tinggi dalam m)^2). (Numerik)
* **DiabetesPedigreeFunction:** Fungsi silsilah diabetes (skor kemungkinan diabetes berdasarkan riwayat keluarga). (Numerik)
* **Age:** Usia (tahun). (Numerik)
* **Outcome:** Variabel target kelas (0 = tidak diabetes, 1 = diabetes). (Numerik, Kategorikal Biner)

### Analisis Data Awal (Exploratory Data Analysis - EDA)

* **Informasi Dasar & Statistik:** Dataset awal memiliki 768 entri dan 9 kolom tanpa nilai null eksplisit (NaN). Namun, analisis nilai 0 pada kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` menunjukkan adanya nilai 0 yang tidak logis secara medis: Glucose (5), BloodPressure (35), SkinThickness (227), Insulin (374), BMI (11). Ini mengindikasikan kemungkinan data hilang yang perlu ditangani. Deskripsi statistik menunjukkan variasi rentang nilai antar fitur dan nilai minimum 0 pada fitur-fitur tersebut, mengindikasikan perlunya penanganan data nol dan scaling. Rata-rata usia pasien sekitar 33 tahun, dengan 35% pasien terdiagnosis diabetes.
* **Duplikasi Data:** Tidak ditemukan baris data yang duplikat.
* **Distribusi Fitur:** Visualisasi histogram menunjukkan bahwa beberapa variabel seperti `Insulin` dan `SkinThickness` memiliki banyak nilai nol dan distribusi yang sangat miring (skewed), mengindikasikan kemungkinan adanya missing value terselubung. Fitur `Glucose`, `BloodPressure`, dan `BMI` tampak lebih mendekati distribusi normal setelah penanganan nilai nol.
* **Outliers:** Box plot menunjukkan adanya beberapa outlier pada beberapa fitur, terutama `Insulin`, `Glucose`, dan `DiabetesPedigreeFunction`, yang memiliki nilai ekstrem jauh di atas rentang normalnya. Penanganan outlier diperlukan untuk algoritma yang sensitif seperti KNN.
* **Distribusi Target:** Terdapat ketidakseimbangan kelas, dengan lebih banyak data untuk kelas 'Tidak Diabetes' (Outcome=0) sebanyak 500 data dibandingkan 'Diabetes' (Outcome=1) sebanyak 268 data. Proporsi kelas 0 sekitar 65% dan kelas 1 sekitar 35%. Ketidakseimbangan ini penting untuk diperhatikan saat evaluasi model (menggunakan metrik seperti Recall dan F1-Score) dan pembagian data (menggunakan stratifikasi).
    ```
    Outcome Distribution:
    0    500
    1    268
    Name: Outcome, dtype: int64
    ```
![image](https://github.com/user-attachments/assets/3c28d67b-86c0-4475-ba53-6e6d3aea2a86)

* **Korelasi Fitur:** Heatmap korelasi menunjukkan korelasi positif yang cukup kuat antara `Glucose` dan `Outcome` (0.47), diikuti oleh `BMI` (0.29), `Age` (0.24), dan `Pregnancies` (0.22). Fitur `BloodPressure`, `SkinThickness`, dan `Insulin` memiliki korelasi yang lebih lemah terhadap `Outcome`. Tidak ada korelasi antar fitur independen yang sangat tinggi (multicollinearity) yang mengharuskan penghapusan fitur.

![image](https://github.com/user-attachments/assets/2c68ace5-c54f-44f9-8364-92ff81b2e2c9)


## Data Preparation

Tahapan persiapan data dilakukan untuk memastikan data siap digunakan oleh model machine learning dan meningkatkan kualitas serta performa model. Langkah-langkah yang dilakukan adalah:

1.  **Penanganan Nilai Tidak Logis (Zeros):** Nilai 0 pada kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` yang dianggap tidak valid secara medis digantikan dengan nilai `NaN` (Not a Number).
    * *Alasan:* Nilai 0 pada fitur-fitur ini umumnya tidak mungkin terjadi pada manusia hidup dan kemungkinan besar mewakili data yang hilang atau tidak tercatat. Menggantinya dengan NaN memungkinkan penanganan missing value yang tepat.
    ```python
    # Contoh kode snippet
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
    ```
    Setelah penggantian, missing values terdeteksi: Glucose (5), BloodPressure (35), SkinThickness (227), Insulin (374), BMI (11).

2.  **Imputasi Missing Values:** Nilai `NaN` yang dihasilkan dari langkah sebelumnya diisi menggunakan nilai **median** dari masing-masing kolom.
    * *Alasan:* Median dipilih karena lebih robust (tidak terlalu terpengaruh) terhadap nilai outlier dan skewness distribusi dibandingkan dengan mean (rata-rata), yang sesuai dengan karakteristik distribusi beberapa fitur dalam dataset ini.
    ```python
    # Contoh kode snippet
    for col in cols_to_replace:
        df[col].fillna(df[col].median(), inplace=True)
    ```
    Setelah imputasi, tidak ada lagi missing values di dataset.

3.  **Penanganan Outliers (IQR Capping):** Outlier pada setiap fitur (kecuali target) ditangani menggunakan metode IQR (Interquartile Range) capping, di mana nilai di luar rentang $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$ diganti dengan batas bawah atau batas atas rentang tersebut.
    * *Alasan:* Outlier dapat sangat mempengaruhi performa beberapa algoritma, terutama yang berbasis jarak atau asumsi normalitas. Capping outlier membantu membatasi pengaruh nilai-nilai ekstrem tanpa menghapusnya sepenuhnya.
    ```python
    # Contoh kode snippet
    # Loop melalui fitur dan terapkan capping
    # ... (detail implementasi capping) ...
    ```
    Setelah capping, boxplot menunjukkan bahwa sebagian besar outlier telah ditangani atau nilainya dibatasi.

4.  **Pemisahan Fitur dan Target:** Dataset dibagi menjadi dua bagian: fitur (variabel independen `X`) yang berisi semua kolom kecuali `Outcome`, dan target (variabel dependen `y`) yang berisi kolom `Outcome`.
    * *Alasan:* Ini adalah langkah standar dalam supervised learning.
    ```python
    # Contoh kode snippet
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    # Shape of X: (768, 8)
    # Shape of y: (768,)
    ```

5.  **Pembagian Data (Train-Test Split):** Dataset (`X` dan `y`) dibagi menjadi data latih (training set) dan data uji (testing set) dengan proporsi 80% untuk latih dan 20% untuk uji. Parameter `stratify=y` digunakan untuk memastikan proporsi kelas target (0 dan 1) sama pada kedua set, menangani masalah ketidakseimbangan kelas. `random_state=42` digunakan untuk reproduktifitas.
    * *Alasan:* Memungkinkan evaluasi objektif pada data tak terlihat. Stratifikasi menjaga representasi kelas target di kedua set.
    ```python
    # Contoh kode snippet
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # X_train shape: (614, 8)
    # X_test shape: (154, 8)
    # y_train shape: (614,)
    # y_test shape: (154,)
    # Distribusi Outcome di train dan test set konsisten dengan distribusi asli
    ```

6.  **Feature Scaling (Standardization):** Fitur-fitur pada data latih dan data uji diskalakan menggunakan `StandardScaler` dari Scikit-learn. Scaler di-`fit` hanya pada data latih dan kemudian digunakan untuk mentransformasi data latih dan data uji.
    * *Alasan:* Standardisasi penting untuk algoritma berbasis jarak seperti KNN. `fit` hanya pada data latih mencegah data leakage.
    ```python
    # Contoh kode snippet
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Data scaled successfully
    ```

## Modeling

Pada tahap ini, model machine learning dibangun dan dilatih menggunakan data yang telah dipersiapkan. Tiga algoritma klasifikasi dipilih dan dibandingkan: K-Nearest Neighbors (KNN), Random Forest, dan Logistic Regression.

1.  **K-Nearest Neighbors (KNN):**
    * **Tahapan:** Model KNN diinisialisasi dengan `n_neighbors=5` sebagai baseline dan dilatih menggunakan `X_train_scaled` dan `y_train`.
    * **Parameter:** `n_neighbors=5` dipilih sebagai nilai awal yang umum digunakan.
    * **Penjelasan Algoritma:** KNN adalah algoritma *lazy learning* yang mengklasifikasikan data baru berdasarkan mayoritas kelas dari 'K' tetangga terdekatnya dalam ruang fitur. Jarak antar titik data (misalnya Euclidean distance) digunakan untuk menentukan kedekatan.
    * **Kelebihan KNN:** Sederhana, mudah diimplementasikan, efektif untuk data dengan pola lokal yang jelas.
    * **Kekurangan KNN:** Sensitif terhadap data dimensi tinggi (curse of dimensionality), sensitif terhadap skala fitur (memerlukan scaling), komputasi bisa mahal saat prediksi pada dataset besar karena perlu menghitung jarak ke semua titik latih.

2.  **Random Forest:**
    * **Tahapan:** Model Random Forest diinisialisasi dengan `n_estimators=100` dan `random_state=42` sebagai baseline dan dilatih menggunakan `X_train_scaled` dan `y_train`.
    * **Parameter:** `n_estimators=100` adalah nilai default yang seringkali memberikan hasil baik. `random_state` memastikan hasil yang sama jika dijalankan ulang.
    * **Penjelasan Algoritma:** Random Forest adalah metode *ensemble learning* yang membangun banyak pohon keputusan (decision trees) selama training. Untuk klasifikasi, outputnya adalah kelas yang paling sering muncul (modus) dari prediksi masing-masing pohon. Algoritma ini menggunakan bagging (bootstrap aggregating) dan pemilihan fitur acak untuk membangun setiap pohon, sehingga mengurangi varians dan overfitting.
    * **Kelebihan Random Forest:** Cenderung sangat akurat, robust terhadap outlier dan data non-linear, menangani data dimensi tinggi dengan baik, memberikan estimasi pentingnya fitur (feature importance), mengurangi risiko overfitting dibandingkan satu pohon keputusan.
    * **Kekurangan Random Forest:** Bisa menjadi 'black box' (sulit diinterpretasikan cara kerjanya secara detail), membutuhkan lebih banyak sumber daya komputasi (memori dan waktu training) dibandingkan algoritma yang lebih sederhana.

3.  **Logistic Regression:**
    * **Tahapan:** Model Logistic Regression diinisialisasi dengan `random_state=42` dan `solver='liblinear'` (cocok untuk dataset kecil) dan dilatih menggunakan `X_train_scaled` dan `y_train`.
    * **Parameter:** `random_state=42`, `solver='liblinear'`.
    * **Penjelasan Algoritma:** Logistic Regression adalah model linear yang memprediksi probabilitas bahwa suatu instance termasuk ke dalam kelas positif (diabetes) menggunakan fungsi logistik.

4.  **Hyperparameter Tuning (Random Forest):**
    * **Tahapan:** Tuning dilakukan pada model Random Forest menggunakan `GridSearchCV` dengan 3-fold cross-validation. Ruang pencarian parameter (`param_grid_rf`) mencakup kombinasi `n_estimators`, `max_depth`, `min_samples_split`, dan `min_samples_leaf`. Metrik yang digunakan untuk optimasi adalah 'accuracy'.
    * **Proses Improvement:** Baseline Random Forest menggunakan parameter default. Tuning bertujuan mencari parameter yang lebih optimal untuk dataset spesifik ini, berpotensi meningkatkan performa dibandingkan baseline.
    * **Parameter Grid:**
        ```python
        param_grid_rf = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 3]
        }
        ```
    * **Hasil Tuning:**
        ```
        Best parameters found for Random Forest:
        {'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 150}
        ```
    * Model terbaik dari hasil tuning (`best_rf`) kemudian disimpan untuk evaluasi.

## Evaluation

Tahap evaluasi bertujuan untuk mengukur performa model yang telah dilatih pada data uji (`X_test_scaled`, `y_test`) yang belum pernah dilihat sebelumnya. Metrik evaluasi yang digunakan sesuai untuk masalah klasifikasi biner ini.

### Metrik Evaluasi

Metrik utama yang digunakan adalah:

1.  **Accuracy:** Proporsi prediksi yang benar (TP + TN) dibagi dengan total jumlah prediksi.
    * *Formula:* $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
    * *Konteks:* Memberikan gambaran umum performa model, namun bisa menyesatkan pada dataset tidak seimbang.
2.  **Precision:** Dari semua prediksi positif (pasien diprediksi diabetes), berapa proporsi yang benar-benar positif?
    * *Formula:* $Precision = \frac{TP}{TP + FP}$
    * *Konteks:* Penting jika biaya False Positive tinggi (misalnya, memberikan pengobatan yang tidak perlu).
3.  **Recall (Sensitivity):** Dari semua kasus positif aktual (pasien benar-benar diabetes), berapa proporsi yang berhasil diprediksi dengan benar oleh model?
    * *Formula:* $Recall = \frac{TP}{TP + FN}$
    * *Konteks:* Sangat penting dalam kasus medis seperti deteksi penyakit. False Negative (FN - pasien diabetes tapi diprediksi tidak) bisa berakibat fatal karena pasien tidak mendapat penanganan. **Recall seringkali menjadi prioritas dalam kasus ini.**
4.  **F1-Score:** Rata-rata harmonik dari Precision dan Recall. Memberikan keseimbangan antara keduanya.
    * *Formula:* $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$
    * *Konteks:* Berguna ketika kita menginginkan keseimbangan antara Precision dan Recall, terutama pada data tidak seimbang.
5.  **Mean Squared Error (MSE):** Rata-rata kuadrat perbedaan antara nilai aktual dan prediksi. Untuk klasifikasi biner 0/1, ini setara dengan proporsi misklasifikasi (1 - Accuracy). Lower MSE lebih baik.
    * *Formula:* $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ (untuk label biner 0/1)
6.  **Confusion Matrix:** Tabel yang menunjukkan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN). Membantu visualisasi performa model secara detail.
7.  **Classification Report:** Ringkasan Precision, Recall, dan F1-Score per kelas, serta Accuracy.

*(TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative)*

### Hasil Evaluasi

Berikut adalah hasil evaluasi pada data uji untuk keempat model (KNN, RF Baseline, RF Tuned, Logistic Regression):

**Ringkasan Performa Model:**

| Model                     | Accuracy | Precision | Recall | F1 Score | MSE    |
| :------------------------ | :------- | :-------- | :----- | :------- | :----- |
| KNN (Baseline)          | 0.7532   | 0.6600    | 0.6111 | 0.6345   | 0.2468 |
| Random Forest (Baseline)  | 0.7662   | 0.6957    | 0.5926 | 0.6400   | 0.2338 |
| Random Forest (Tuned)   | 0.7597   | 0.6889    | 0.5741 | 0.6263   | 0.2403 |
| Logistic Regression       | 0.7143   | 0.6087    | 0.5185 | 0.5600   | 0.2857 |

**1. KNN (Baseline K=5):**
* Accuracy: 0.7532
* Precision (Diabetes, Class 1): 0.6600
* Recall (Diabetes, Class 1): 0.6111
* F1 Score (Diabetes, Class 1): 0.6345
* MSE: 0.2468
* Confusion Matrix:
    * TN (Diprediksi 0, Aktual 0): 83
    * FP (Diprediksi 1, Aktual 0): 17
    * FN (Diprediksi 0, Aktual 1): 21
    * TP (Diprediksi 1, Aktual 1): 33
    Total = 83 + 17 + 21 + 33 = 154 (sesuai jumlah data uji)
* Classification Report (Kelas 1 - Diabetes): Precision 0.66, Recall 0.61, F1-Score 0.63.
  
![image](https://github.com/user-attachments/assets/23b488bd-1924-47ae-a5e8-9ce2dc49cbff)

**2. Random Forest (Baseline):**
* Accuracy: 0.7662
* Precision (Diabetes, Class 1): 0.6957
* Recall (Diabetes, Class 1): 0.5926
* F1 Score (Diabetes, Class 1): 0.6400
* MSE: 0.2338
* Confusion Matrix:
    * TN (Diprediksi 0, Aktual 0): 82
    * FP (Diprediksi 1, Aktual 0): 18
    * FN (Diprediksi 0, Aktual 1): 22
    * TP (Diprediksi 1, Aktual 1): 32
* Classification Report (Kelas 1 - Diabetes): Precision 0.70, Recall 0.59, F1-Score 0.64.

![image](https://github.com/user-attachments/assets/3a92e846-4eca-4471-bc4f-f4a7df91f9e0)

**3. Random Forest (Tuned):**
* Accuracy: 0.7597
* Precision (Diabetes, Class 1): 0.6735
* Recall (Diabetes, Class 1): 0.5741
* F1 Score (Diabetes, Class 1): 0.6200
* MSE: 0.2403
* Confusion Matrix:
    * TN (Diprediksi 0, Aktual 0): 81
    * FP (Diprediksi 1, Aktual 0): 19
    * FN (Diprediksi 0, Aktual 1): 23
    * TP (Diprediksi 1, Aktual 1): 31
* Classification Report (Kelas 1 - Diabetes): Precision 0.67, Recall 0.57, F1-Score 0.62.

![image](https://github.com/user-attachments/assets/7af08cd3-fdce-429c-9508-09b457ce7a2b)

**4. Logistic Regression:**
* Accuracy: 0.7143
* Precision (Diabetes, Class 1): 0.6087
* Recall (Diabetes, Class 1): 0.5185
* F1 Score (Diabetes, Class 1): 0.5600
* MSE: 0.2857
* Confusion Matrix:
    * TN (Diprediksi 0, Aktual 0): 82
    * FP (Diprediksi 1, Aktual 0): 18
    * FN (Diprediksi 0, Aktual 1): 26
    * TP (Diprediksi 1, Aktual 1): 28
* Classification Report (Kelas 1 - Diabetes): Precision 0.61, Recall 0.52, F1-Score 0.56.

![image](https://github.com/user-attachments/assets/d6b6a5b7-f6e7-4985-bb69-0d9818fb1764)

### Menjabarkan hasil evaluasi
Dalam proyek ini, model machine learning yang dikembangkan telah berhasil menjawab seluruh problem statement yang dirumuskan. Pertama, proses pembangunan model dilakukan dengan menerapkan dan membandingkan tiga algoritma klasifikasi yang berbeda, yaitu K-Nearest Neighbors (KNN), Random Forest, dan Logistic Regression, terhadap dataset yang telah diproses. Hal ini memungkinkan untuk mengevaluasi efektivitas masing-masing algoritma dalam memprediksi keberadaan diabetes pada pasien berdasarkan fitur diagnostik yang tersedia. Kedua, pertanyaan mengenai algoritma mana yang memberikan performa terbaik juga telah dijawab melalui evaluasi menggunakan metrik akurasi, presisi, recall, F1-score, dan MSE. Hasil evaluasi menunjukkan bahwa model Random Forest (Baseline) memberikan performa terbaik secara keseluruhan, dengan akurasi tertinggi sebesar 76,62%, precision sebesar 69,57%, F1-score tertinggi sebesar 64,00%, dan MSE terendah sebesar 0,2338. Meskipun KNN memiliki recall sedikit lebih tinggi (61,11%), keunggulan metrik lain pada Random Forest menjadikannya pilihan yang lebih seimbang dan optimal.

Selain itu, tujuan utama proyek juga tercapai dengan baik. Model yang dibangun mampu mengklasifikasikan apakah seorang pasien menderita diabetes (Outcome = 1) atau tidak (Outcome = 0) dengan tingkat akurasi dan recall yang cukup tinggi. Perbandingan antar-algoritma pun telah dilakukan secara menyeluruh, sesuai dengan tujuan untuk menemukan model yang paling optimal berdasarkan metrik evaluasi yang relevan. Dengan demikian, seluruh solusi yang dirancang—mulai dari penerapan tiga algoritma klasifikasi, penggunaan metrik evaluasi yang komprehensif, hingga analisis yang mempertimbangkan konteks medis seperti pentingnya recall dalam diagnosis penyakit—telah memberikan dampak nyata terhadap pemahaman bisnis (business understanding) dari masalah ini. Evaluasi yang mendalam ini memastikan bahwa model yang dipilih tidak hanya optimal secara teknis, tetapi juga relevan dan berkontribusi langsung terhadap pengambilan keputusan dalam konteks prediksi diabetes.

### Analisis Hasil

* **Perbandingan Umum:** Model Random Forest (Baseline) dan KNN (Baseline) menunjukkan performa yang lebih baik dibandingkan Random Forest (Tuned) dan Logistic Regression berdasarkan sebagian besar metrik. Logistic Regression adalah model dengan performa terendah.
* **Berdasarkan F1-Score:** Random Forest (Baseline) memiliki F1-Score tertinggi (0.6400), menunjukkan keseimbangan terbaik antara Precision dan Recall di antara semua model.
* **Berdasarkan Recall:** KNN (Baseline) memiliki Recall tertinggi (0.6111), yang berarti model ini paling baik dalam mengidentifikasi kasus diabetes aktual, meskipun sedikit lebih rendah dalam Precision dibandingkan Random Forest Baseline.
* **Berdasarkan Precision:** Random Forest (Baseline) memiliki Precision tertinggi (0.6957) untuk kelas diabetes, artinya ketika model ini memprediksi diabetes, kemungkinannya benar paling tinggi.
* **MSE:** Random Forest (Baseline) memiliki MSE terendah (0.2338), yang konsisten dengan akurasi tertingginya karena MSE untuk klasifikasi biner adalah 1 - Accuracy.
* **Dampak Hyperparameter Tuning:** Tuning hyperparameter pada Random Forest dengan grid dan validasi silang yang digunakan ternyata tidak meningkatkan performa pada data uji; justru sedikit menurunkan Akurasi, Precision, Recall, dan F1-Score dibandingkan model baseline. Ini menunjukkan bahwa parameter default atau parameter lain di luar ruang pencarian yang dicoba mungkin lebih optimal untuk dataset ini, atau validasi silang 3-fold tidak cukup representatif.
* **Jenis Kesalahan (Confusion Matrix):** Semua model memiliki sejumlah False Negatives (FN), yang berarti ada kasus diabetes yang tidak terdeteksi. Jumlah FN tertinggi ada pada Logistic Regression (26), diikuti Random Forest Tuned (23), Random Forest Baseline (22), dan terendah pada KNN Baseline (21). Mengurangi FN adalah prioritas penting dalam deteksi medis. Semua model juga memiliki False Positives (FP), yang berarti memprediksi diabetes pada pasien yang sebenarnya tidak menderita diabetes. Jumlah FP berkisar antara 17 hingga 19.

**Kesimpulan Evaluasi:** Model **Random Forest (Baseline)** menunjukkan performa paling baik secara keseluruhan berdasarkan F1-Score (0.6400), Accuracy (0.7662), Precision (0.6957), dan MSE terendah (0.2338). Model ini memberikan keseimbangan yang baik antara mengidentifikasi kasus diabetes dan memastikan keakuratan prediksi positifnya. Namun, jika prioritas mutlak adalah *meminimalkan kasus diabetes yang tidak terdeteksi (False Negatives)*, model **KNN (Baseline)** bisa dipertimbangkan karena memiliki Recall tertinggi (0.6111), meskipun dengan trade-off sedikit penurunan pada metrik lain seperti Precision dan F1-Score.

Meskipun tuning Random Forest dilakukan, pada data uji ini, model baseline memberikan hasil yang sedikit lebih baik. Ini menunjukkan bahwa pemilihan parameter optimal sangat bergantung pada data spesifik dan validasi yang kuat diperlukan. Model Logistic Regression memiliki performa terendah di antara algoritma yang dicoba.

Oleh karena itu, berdasarkan hasil evaluasi ini, model Random Forest (Baseline) atau KNN (Baseline) adalah kandidat terbaik untuk solusi prediksi diabetes menggunakan dataset ini, tergantung pada apakah prioritasnya adalah keseimbangan performa secara umum (F1-Score) atau meminimalkan False Negative (Recall).
