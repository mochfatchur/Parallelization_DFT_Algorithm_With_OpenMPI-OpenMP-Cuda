# Penjelasan Cara Kerja Parelisasi
Parelisasi yang dibuat adalah sebagai berikut :
Eksekusi blok program for loop (untuk perhitungan freq_domain.mat dan penjumlahan sum) 
dibagikan ke n threads (misal num_threads(8) berarti dibagikan ke 8 thread), rincian pembagian adalah sebagai berikut : 
- indeks loop k yang sama memiliki thread yang sama 
- misalnya : (k=0,l = {0,1,...} ==> dieksekusi oleh thread 0), (k=1, l = {0,1,..} ==> dieksekusi oleh thread 1), dst
- Kemudian hasil dari sum masing-masing thread akan di reduksi/digabungkan

# Penjelasan Cara Program Membagikan Data Antar Thread
Cara Program Membagikan Data Antar Proses adalah sebagai berikut:
- Data Matrix source yang dibaca akan bersifat shared (artinya bisa diakses oleh semua thread didalam team thread)
- variabel seperti indeks loop k dan l bersifat private(hanya bisa diakses oleh 1 thread, namun tiap thread punya salinan variabel tersebut)
- Masing-masing thread akan memiliki versi sum nya sendiri, kemudian nantinya akan di reduksi menjadi sum total

# Alasan Pemilihan Skema Pembagian Data
Alasan dilakukan pembagian data seperti itu:
- Data matrix harus bisa dibaca oleh semua thread karena setiap thread melakukan eksekusi freq_domain.mat[k][l] = dft(&source, k, l) dan double complex el = freq_domain.mat[k][l] 
  yang mana ini memerlukan source yang bersifat shared
- Data indeks loop k dan l bersifat private karena masing-masing thread hanya perlu k dan l versi mereka sendiri, tanpa peduli perubahan k dan l thread lainnya
- Variabel sum secara implisit bersifat private, proses penjumlahan untuk mendapatkan nilai sum bisa dilakukan secara pararel karena tidak ada dependency dari proses lainnya
