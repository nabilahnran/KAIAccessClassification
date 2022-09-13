# KAI Access Review Sentiment and Topic Classification

## Project of PT Kereta Api Indonesia Intership Program
(have permission to public this project)

Nabiilah Nuur Ainii, Islamic University of Indonesia (2022)

Proyek ini adalah salah satu proyek dalam kegiatan magang yang dilakukan di PT Kereta Api Indonesia pada September 2021 - Maret 2022.

  KAI Access adalah satu-satunya aplikasi resmi yang dikeluarkan oleh PT Kereta Api Indonesia untuk penjualan tiket kereta api secara daring. Aplikasi ini tidak hanya dibuat untuk penjualan tiket, beberapa fitur telah ditambahkan untuk kemudahan dan pelayanan kepada customer PT Kereta Api Indonesia (KAI, 2021). Banyak transaksi yang dilakukan pada aplikasi ini, terutama semenjak pandemi Covid-19 berlangsung yang mengharuskan calon penumpang melakukan segala transaksi tiket kereta api secara daring yang salah satunya melalui aplikasi KAI Access. Pemesanan dan pembelian tiket, pengubahan jadwal keberangkatan, pembatalan tiket, transaksi KA Logistik, pemesanan makanan dan minuman di dalam kereta api, hingga top up dan pembayaran tagihan dengan e-wallet KAI Access yaitu KAIPay dapat dilakukan dalam aplikasi ini.

  Penggunaan aplikasi yang terus meningkat sehingga diperlukannya peningkatan kualitas dari aplikasi ini untuk meningkatkan kepuasan pengguna. Oleh karena itu dengan memanfaatkan data ulasan KAI Access pada platform distribusi Play Store, penulis mengembangkan sistem klasifikasi sentimen dan topik data. Sistem klasifikasi sentimen dan topik data ulasan KAI Access ini menggunakan natural language processing dengan metode deep learning-neural networks yang akan mempelajari data teks ulasan yang ditulis oleh pengguna.
  
  Beberapa tools dipakai oleh penulis dalam pengerjaan proyek ini yaitu Google Colab, PyCharm (Python), Google Sheet, dan framework machine learning TensorFlow. Pada proses preprocessing data juga memakai beberapa library Python seperti NumPy, Sklearn, dan Pandas. Penulis memakai metode pembelajaran mesin deep learning dengan algoritma Recurrent Neural Network dan Bidirectional Long Short Term Memory (BiLSTM) untuk model proyek ini dengan menggunakan TensorFlow. Data ulasan KAI Access juga akan dianalisis dan divisualisasikan dalam bentuk dashboard menggunakan tools Tableau.

  Hasil dari proyek KAI Access ini adalah data yang dapat diklasifikasikan secara langsung dengan machine learning. Klasifikasi pada proyek ini dilakukan untuk 3 label yaitu sentimen, topik, dan detail topik. Hal ini dikarenakan proyek KAI Access membutuhkan hasil klasifikasi topik yang lebih detail untuk dapat mengidentifikasi keluhan pengguna pada sistem dengan spesifik lebih cepat. Selain klasifikasi, proyek ini juga menghasilkan insight dari analisis yang dilakukan dari data ulasan KAI Access ini. Insight yang divisualisasikan seperti jumlah data ulasan perbulan, versi yang paling banyak dipakai, jumlah dan rata-rata rating penilaian, dan jumlah data setiap kelas pada label sentimen, topik, dan detail topik.
  
**Gambaran Prototype Klasifikasi Data Ulasan KAI Access
(Proses pemasukan file)
<img width="434" alt="image" src="https://user-images.githubusercontent.com/57905354/189805350-4aa752f8-c397-467a-91ec-b6a87a976602.png">
(Hasil klasifikasi)
<img width="348" alt="image" src="https://user-images.githubusercontent.com/57905354/189805362-ec3b5e1b-a5ef-4a20-90dc-f257bf404194.png">

  
*Dokumen yang dicantumkan pada github hanya code pengembangan prototype sistem klasifikasi otomatis, pengembangan model bersifat privat.
