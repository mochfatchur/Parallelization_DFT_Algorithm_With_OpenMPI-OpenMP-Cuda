# Penjelasan Cara Kerja Parelisasi
Parelisasi yang dibuat adalah sebagai berikut :
- Rank 0 akan melakukan ReadMatrix dan lalu akan melakukan Broadcast sehingga dapat diakses oleh proses-proses lain (semua rank)  
- Satu proses menangani sejumlah (size_matrix / world_size) baris dan baris yang ditangani pada row = (rank + k * world_size)
- (misal pada size_matrix=32, world_size=4 proses 0 menangani baris 0, 4, 8, 12, 16 dll)
- (misal pada size_matrix=32, world_size=4 proses 1 menangani baris 1, 5, 9, 13, 17 dll)
- Setiap proses melakukan dft pada elemen matrix pada baris tersebut dan menjumlahkannya yang akan disimpan sebagai local_sum
- Jika rank selain 0 akan mengirim ke rank 0:
  - local_sum ke rank 0, 
  - bilangan realnya dikirim dengan tag=0, 
  - dan bilangan imaginer dengan tag=1
- Jika rank 0:
  - Menjumlahkan semua local_sum dari prosesnya sendiri (rank 0) dan local_sum dari rank lain
  - Mencetak elemen matrix hasil dft pada baris 0 saja dan average dari matrix

# Penjelasan Cara Program Membagikan Data Antar Proses
Cara Program Membagikan Data Antar Proses adalah sebagai berikut:
- Data Matrix yang dibaca di Broadcast  sehingga dapat diakses oleh proses-proses lain (semua rank)
- Satu proses menangani sejumlah (size_matrix / world_size) baris dan baris yang ditangani adalah pada row = (rank + k * world_size)
- Proses tersebut melakukan local_sum, yang akan di kirimkan ke rank 0
- (Misal pada size_matrix=32, world_size=4 proses 0 menangani baris 0, 4, 8, 12, 16 dll)
- (Misal pada size_matrix=32, world_size=4 proses 1 menangani baris 1, 5, 9, 13, 17 dll)

# Alasan Pemilihan Skema Pembagian Data
Alasan dilakukan pembagian data per baris karena tidak ada dependensi antar barisnya sehingga dapat di parelisasi.

