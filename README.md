# Laporan Proyek Machine Learning - Nurul Nyi Qoniah

## Domain Proyek

Membuat model predictive analytics (housing price prediction) menggunakan dataset dari Kaggle [Housing Dataset](https://www.kaggle.com/code/abdelrahmanramadan2/housing-price-prediction-using-linear-regression) sebagai salah satu submission Dicoding. Pada proyek ini akan diprediksi harga rumah dengan menggunakan dataset yang tersedia sebelumnya.

### Latar Belakang

Proyek ini bertujuan untuk membandingkan kinerja tiga algoritma machine learning yang berbeda, yaitu K-Nearest Neighbors (KNN), Adaboost, dan Random Forest, dalam memprediksi harga rumah. Prediksi harga rumah memiliki signifikansi penting dalam industri real estate, membantu pemilik properti, pembeli, dan pemasok informasi pasar untuk membuat keputusan yang tepat terkait investasi dan transaksi properti.

Metode KNN adalah pendekatan berbasis instance yang menghitung jarak antara titik data untuk menentukan label kelasnya. Adaboost adalah algoritma pembelajaran ensemble yang menggabungkan beberapa model lemah menjadi satu model kuat dengan memberikan bobot yang lebih tinggi pada sampel yang salah diklasifikasikan sebelumnya. Sementara itu, Random Forest adalah algoritma pembelajaran ensemble yang membangun beberapa pohon keputusan dan menggabungkan hasil prediksi mereka.

Proyek ini menggunakan dataset harga rumah yang luas dan beragam fitur seperti luas tanah, jumlah kamar, fasilitas, dan lokasi. Data tersebut kemudian dibagi menjadi subset pelatihan dan pengujian untuk mengevaluasi kinerja setiap algoritma dalam memprediksi harga rumah.

Hasil proyek ini diharapkan dapat memberikan wawasan yang berharga tentang keunggulan dan kelemahan masing-masing algoritma dalam memprediksi harga rumah. Diharapkan pula hasilnya dapat membantu pemangku kepentingan di industri real estate dalam mengambil keputusan yang lebih baik berdasarkan prediksi harga rumah yang lebih akurat.

![Rumah](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/rumah.png)

Gambar 1. Rumah 


## Business Understanding

Tujuan dari proyek ini adalah untuk mengembangkan model machine learning yang dapat memprediksi harga rumah dengan tingkat akurasi yang tinggi. Dengan melakukan analisis data yang cermat dan menggunakan teknik machine learning yang sesuai, diharapkan bahwa model yang dihasilkan dapat memberikan wawasan yang berharga bagi pemangku kepentingan di pasar properti.

### Problem Statements

- Bagaimana cara melakukan data preparation agar dapat digunakan dalam model machine learning?
- Bagaimana cara menentukan nilai target variable tersebut?
- Algoritma apa yang paling baik dalam menentukan harga rumah?

### Goals

- Dapat melakukan analisa data dan pengolahan data agar dapat digunakan oleh model machine learning
- Dapat membuat model yang dapat memprediksi harga rumah.
- Mengetahui algoritma yang paling efektif untuk mendata dan melakukan prediksi harga rumah untuk membantu pihak perusahaan dalam menentukan harga rumah yang sesuai.

### Solution statements

- Menganalisis data dengan melakukan EDA atau Exploratory Data Analysis untuk mengetahui informasi tentang dataset dengan melakukan univariate dan multivariate analysis, dan visualisasi data.
- Membangun model machine learning untuk bisa memprediksi harga rumah dengan menggunakan 3 algoritma machine learning yaitu K-Nearest Neighbour, Random Forest, dan AdaBoost.
- Melakukan hyperparameter tuning untuk mendapatkan nilai parameter terbaik sehingga bisa mendapatkan model terbaik dengan menggunakan teknik Grid Search.

## Data Understanding dan Exploratory Data Analysis

Dataset yang digunakan dalam proyek ini merupakan data harga penjualan rumah. Dataset ini dapat diunduh di Contoh: [Housing Dataset](https://www.kaggle.com/datasets/ashydv/housing-dataset/data).

Berikut informasi pada dataset :

- Dataset dengan format CSV (Comma-Seperated Values)
- Dataset memiliki 545 record dengan 13 feature
- Dataset memiliki 6 feature numerik dan 7 feature kategori
- Tidak terdapat missing value dalam dataset

### Variabel-variabel pada Housing dataset adalah sebagai berikut:

- price : merupakan harga rumah yang akan dijual
- area : merupakan luas rumah yang tersedia
- bedrooms : merupakan banyak kamar tidur yang tersedia
- bathrooms : merupakan banyak toilet yang tersedia
- stories : merupakan banyak tingkatan rumah yang ada
- mainroad : merupakan jenis jalan utama yang terdekat dengan tempat tinggal
- guestroom : merupakan status rumah apakah memiliki kamar tamu atau tidak
- basement : merupakan status rumah apakah memiliki basement atau tidak
- hotwaterheating : merupakan status rumah apakah memiliki pemanas air atau tidak
- airconditioning : merupakan status rumah apakah memiliki air conditioning atau tidak
- parking : merupakan status rumah apakah memiliki tempat parkir atau tidak
- prefarea : merupakan status rumah apakah disukai atau tidak
- furnishingstatus : merupakan status rumah apakah Furnished atau Semi-Furnished atau Unfurnished.

### Univariate Analysis

Analisis univariat adalah jenis analisis statistik yang dilakukan pada satu variabel tunggal dalam sebuah dataset. Tujuan dari analisis univariat adalah untuk memahami karakteristik atau distribusi dari variabel tersebut secara terpisah, tanpa memperhatikan hubungan dengan variabel lain dalam dataset.

terdapat 2 jenis fitur yaitu fitu numerik dan fitur kategori

**Fitur numerik**
![Numerik](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/numerik.png)

Gambar 2. Fitur numerik
**Fitur kategorikal**

Mainroad

![Mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/mainroad.png)

Gambar 3. Univariate mainroad

Guest Room

![Guest Room](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/guestroom.png)

Gambar 4. Univariate Guest Room

Basement

![Basement](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/basement.png)

Gambar 5. Univariate basement

Hot Water Heating

![Hot Water Heating](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/hotwater.png)

Gambar 6. Univariate hot water heating

Air Conditioning

![Air Conditioning](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/ac.png)

Gambar 7. Univariate ac

Preferend Area

![Preferend Area](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/pref.png)

Gambar 8. Univariate prefend area

### Multivariate Analysis

Analisis multivariat adalah jenis analisis statistik yang dilakukan pada dua atau lebih variabel dalam sebuah dataset. Tujuan dari analisis multivariat adalah untuk memahami hubungan kompleks antara variabel-variabel tersebut dan mengidentifikasi pola atau struktur yang mungkin tersembunyi di antara mereka.

Dalam analisis multivariat, data dieksplorasi dan dianalisis untuk memahami hubungan antara variabel-variabel tersebut.

**Fitur numerik**
![Multivariate Numerik](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/multivariatenumerik.png)

Gambar 9. Multivariate numerik

Fitur Price dengan mainroad
![Fitur Price dengan mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/mainprice.png)

Gambar 10. Multivariate price dengan mainroad

Fitur Price dengan guestroom
![Fitur Price dengan mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/guestroomprice.png)

Gambar 11. Multivariate price dengan guest room 

Fitur Price dengan basement
![Fitur Price dengan mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/basementprice.png)

Gambar 12. Multivariate price dengan basement

Fitur Price dengan hot water heater
![Fitur Price dengan mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/hotwaterprice.png)

Gambar 13. Multivariate price dengan hot water heater

Fitur Price dengan air conditioning
![Fitur Price dengan mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/acprice.png)

Gambar 14. Multivariate price dengan air conditioning

Fitur Price dengan preferend area
![Fitur Price dengan mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/prefprice.png)

Gambar 15. Multivariate price dengan prefend area

Fitur Price dengan furnishing status
![Fitur Price dengan mainroad](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/furprice.png)

Gambar 16. Multivariate price dengan furnishing status

## Data Preparation

Persiapan data (data preparation) adalah tahap penting dalam analisis data yang melibatkan persiapan dataset agar siap digunakan untuk analisis lebih lanjut.

### Encoding Fitur Kategori

Encoding fitur kategori adalah proses mengubah variabel kategori menjadi bentuk yang dapat diproses oleh algoritma machine learning, yang umumnya memerlukan input numerik.

| index | price   | area | bedrooms | bathrooms | stories | mainroad | guestroom | basement | hotwaterheating | airconditioning | parking | prefarea | furnished | semi-furnished |
| ----- | ------- | ---- | -------- | --------- | ------- | -------- | --------- | -------- | --------------- | --------------- | ------- | -------- | --------- | -------------- |
| 15    | 9100000 | 6000 | 4        | 1         | 2       | 1        | 0         | 1        | 0               | 0               | 2       | 0        | 0         | 1              |
| 16    | 9100000 | 6600 | 4        | 2         | 2       | 1        | 1         | 1        | 0               | 1               | 1       | 1        | 0         | 0              |
| 17    | 8960000 | 8500 | 3        | 2         | 4       | 1        | 0         | 0        | 0               | 1               | 2       | 0        | 1         | 0              |
| 18    | 8890000 | 4600 | 3        | 2         | 2       | 1        | 1         | 0        | 0               | 1               | 2       | 0        | 1         | 0              |
| 19    | 8855000 | 6420 | 3        | 2         | 2       | 1        | 0         | 0        | 0               | 1               | 1       | 1        | 0         | 1              |

Tabel 1. Hasil Encoding feeture kategori

### Train-Test-Split

Train-test-split adalah teknik yang umum digunakan dalam machine learning untuk membagi dataset menjadi subset pelatihan (training set) dan subset pengujian (test set).

![train test](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/tts.png)

Gambar 17. Train test split

### Standarisasi

Standarisasi mengubah skala data sehingga memiliki mean nol dan deviasi standar satu

| index | area                 | bedrooms             | bathrooms            | stories              | parking             |
| ----- | -------------------- | -------------------- | -------------------- | -------------------- | ------------------- |
| 136   | 0\.30496363758601286 | 1\.444648919726656   | 1\.575019269668501   | 0\.23297192992872032 | 1\.6151370978060522 |
| 534   | -1\.0679097021718376 | 1\.444648919726656   | -0\.5602109355672222 | 0\.23297192992872032 | -0\.77290344426013  |
| 322   | -0\.8047756453849163 | 0\.0792795138874385  | 1\.575019269668501   | -0\.9073643586697525 | 0\.4211168267729612 |
| 30    | 1\.491927045918321   | 0\.0792795138874385  | 1\.575019269668501   | 2\.5136445071256657  | 1\.6151370978060522 |
| 354   | 2\.021055312283326   | -1\.2860898919517791 | -0\.5602109355672222 | -0\.9073643586697525 | 0\.4211168267729612 |

Tabel 2. Hasil standarisasi feature numerik

## Modeling

Algoritma Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan Boosting

### K-Nearest Neighbour

K-Nearest Neighbors (KNN) adalah salah satu algoritma pembelajaran mesin yang paling sederhana dan mudah dipahami. Prinsip dasar dari KNN adalah memprediksi kelas atau nilai target suatu observasi berdasarkan mayoritas kelas dari K observasi terdekat di antara data latihnya.

Dalam algoritma ini digunakan library scikit-learn untuk membuat model regresi berbasis K-Nearest Neighbors (KNN) .

```
from sklearn.neighbors import KNeighborsRegressor
```

terdapat parameter `n_neighbors` : jumlah neighbor yang digunakan untuk melakukan prediksi, biasanya diatur ke nilai maksimum.

**kelebihan**
+ Sederhana: Mudah dipahami dan diimplementasikan.
+ Sedikit parameter
+ Adaptif terhadap data baru: Dapat dengan cepat memperbarui modelnya dengan data baru.

**kekurangan**
+ Lambat pada dataset besar karena perlu menghitung jarak dari setiap titik data ke semua titik lainnya.
+ Rentan terhadap pengaruh data pencilan.
+ Perlu penskalaan fitur dan pemilihan parameter k yang tepat.
+ Kurang efektif pada dataset dengan jumlah fitur yang tinggi.

### Random Forest

Random Forest adalah algoritma pembelajaran mesin yang termasuk dalam kategori ensemble learning. Ensemble learning adalah teknik yang menggabungkan beberapa model pembelajaran mesin untuk meningkatkan kinerja dan stabilitas prediksi. Random Forest menggabungkan konsep "bagging" dengan pohon keputusan.

Dalam algoritma ini digunakan library scikit-learn untuk membuat model regresi berbasis Random Forest (RF).

```
from sklearn.ensemble import RandomForestRegressor
```

terdapat parameter beberapa parameter sebagai berikut:

- `n_estimators` : yang menentukan jumlah tree yang dibuat oleh RF.
- `max_depth`: kedalaman maksimum setiap pohon keputusan dalam ensemble. Untuk mengontrol kompleksitas setiap pohon dan dapat membantu mencegah overfitting.
- `random_state` : untuk menetapkan keadaan acak, sehingga hasil pembangunan model akan konsisten jika dijalankan berulang kali.
- `n_jobs` : menentukan jumlah pekerjaan yang akan dieksekusi secara paralel saat melatih model.

**kelebihan**
+ Random Forest dapat menangani dataset dengan kelas yang tidak seimbang dengan baik, karena menerapkan voting mayoritas untuk klasifikasi.
+ Random Forest tidak memiliki banyak parameter untuk disetel, sehingga mudah diimplementasikan dan digunakan.

**kekurangan**
+ Lebih sulit untuk diinterpretasikan daripada model linear sederhana karena keberagaman pohon keputusan yang terlibat.
+ Random Forest bisa memerlukan sumber daya komputasi yang cukup besar, terutama pada dataset yang besar dengan banyak pohon atau fitur.
+ Jika data memiliki ketergantungan yang kuat antar fitur, Random Forest mungkin tidak optimal karena tidak mampu menangkap pola-pola kompleks seperti halnya model yang lebih fleksibel.

### Boosting

Boosting adalah teknik ensemble learning lainnya yang bekerja dengan cara menggabungkan sejumlah model pembelajaran mesin yang lemah (weak learner) menjadi satu model yang kuat (strong learner). Prinsip dasar dari boosting adalah mempelajari sekumpulan model secara berurutan, di mana setiap model mencoba untuk memperbaiki kesalahan prediksi model sebelumnya.

Dalam algoritma ini digunakan library scikit-learn untuk membuat model regresi berbasis AdaBoost

```
from sklearn.ensemble import AdaBoostRegressor
```

terdapat parameter beberapa parameter sebagai berikut:

- `learning_rate` : untuk memperbarui bobot pada setiap iterasi.
- `random_state` : untuk menetapkan keadaan acak, sehingga hasil pembangunan model akan konsisten jika dijalankan berulang kali

**kelebihan**
+ Sering menghasilkan model yang memiliki kinerja yang sangat baik
+ Mengatasi Overfitting: Dengan fokus pada sampel yang salah diklasifikasikan pada iterasi sebelumnya, Adaboosting cenderung mengurangi overfitting, terutama jika digunakan dengan model dasar yang sederhana.
+ Adaboosting efektif dalam menangani dataset dengan kelas yang tidak seimbang karena memberikan bobot yang lebih tinggi pada sampel yang salah diklasifikasikan.

**kekurangan**
+ Sensitif terhadap Noise dan Outliers
+ Membutuhkan Waktu Pembelajaran yang Lama
+ Tidak Cocok untuk Data dengan Banyak Fitur

### Hyperparameter Tuning

Hyperparameter tuning adalah proses mengambil nilai terbaik dari hyperparameter yang tidak langsung diketahui

Untuk melakukan hyperparameter tuning, kita dapat menggunakan sklearn.model_selection.GridSearchCV. Maka dapat dihasilkan parameter terbaik sebagai berikut
|index|model|best_score|best_params|
|---|---|---|---|
|0|knn|0\.3753532280060405|\{'n_neighbors': 5\}|
|1|random_forest|0\.6683750141592485|\{'max_depth': 16, 'n_estimators': 100, 'n_jobs': 1, 'random_state': 33\}|
|2|boosting|0\.5742617820131491|\{'learning_rate': 0\.1, 'random_state': 11\}|

Tabel 3. Hasil hyperparameter tuning

## Evaluation

Evaluasi model yang digunakan adalah mean squared error (MSE). MSE adalah singkatan dari Mean Squared Error. Ini adalah salah satu metrik evaluasi yang umum digunakan dalam masalah regresi untuk mengukur seberapa baik model memprediksi nilai yang kontinu. Secara matematis, MSE dapat dihitung dengan rumus berikut: 

MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$

dimana: 
+ n = jumlah titik data
+ Yi = nilai sesungguhnya
+ Yi_hat = nilai prediksi

MSE mengukur rata-rata dari kuadrat perbedaan antara prediksi model dan nilai sebenarnya. Semakin rendah nilai MSE, semakin baik model dalam memprediksi nilai sebenarnya. Sebaliknya, nilai MSE yang lebih tinggi menunjukkan bahwa model cenderung melakukan prediksi yang buruk.

Hasil dari evaluasi tersebut adalah:
  
|index|train|test|
|---|---|---|
|KNN|682149066\.1567054|5724387873\.076922|
|RF|135118786\.1627713|4494885131\.828696|
|Boosting|897755190\.600389|3580515918\.0956063|

Tabel 4. Hasil evaluasi matriks mse

![alt text](https://github.com/nurqoneah/Predictive-Analysis-Housing/blob/main/mse.png)

Gambar 18. evaluasi matriks mse

Hasil Prediksi
|index|y_true|prediksi_KNN|prediksi_RF|prediksi_Boosting|
|---|---|---|---|---|
|428|3325000|2667000\.0|2342386\.7|3160166\.2|

Tabel 5. Hasil prediksi 3 algoritma

Dari hasil evaluasi di atas dapat kita lihat bahwa Algoritma AdaBoost Regression yang memiliki akurasi terbaik dan loss terkecil dibanding algiritma lainnya.


## Referensi 
+ Putri, V. A. P., Prasetijo, A. B., & Eridani, D. (2022). Perbandingan Kinerja Algoritme Naïve Bayes dan K-Nearest Neighbor (KNN) untuk Prediksi Harga Rumah. Transmisi: Jurnal Ilmiah Teknik Elektro, 24(4), 162-171. https://ejournal.undip.ac.id/index.php/transmisi/article/view/47129/0
+ Prianti, A. I., Santoso, R., & Hakim, A. R. (2020). Perbandingan Metode K-Nearest Neighbor dan Adaptive Boosting pada Kasus Klasifikasi Multi Kelas. Gaussian, 9(3), 346-354. https://ejournal3.undip.ac.id/index.php/gaussian/article/download/28924/24520

