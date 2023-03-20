# Penjelasan Cara Kerja Parelisasi
Parelisasi yang dibuat adalah sebagai berikut :
- Bagian yang akan dikerjakan di GPU/devices adalah bagian eksekusi program dft dan penjumlahan elemen freq_domain
- dimensi block yang digunakan adalah (16,16) artinya terdapat 256 thread, dengan dimensi gridnya adalah (ceil(source.size/block.x) , ceil(source.size/block.y))
- dengan melakukan eksekusi program dft secara paralel dengan cuda dapat mengoptimasi eksekusi yang awalnya O(N^4) menjadi O(N^2) karena setiap thread akan lagsung melakukan dft pada row dan column tertentu.


# Penjelasan Cara Program Membagikan Data
data source matrix, freq_domain matrix dan sum akan dialokasikan ke GPU/devices kemudian setiap thread akan ada versi sum sementaranya (secara implisit) yang  nantinya akan dilakukan atomicAdd untuk memperoleh hasil sum total global, kemudian hasil sum dan freq_domain matrix yang telah terisi elemen yang telah ter-dft akan dikirimkan ke host.

# Alasan Pemilihan Skema Pembagian Data
Data yang dikirimkan ke devices/GPU untuk sources dan freq_domain matrix tidak ada yang konflix, setiap elemen matrix freq_domain yang telah ter-dft pada row dan column tertentu langsung dijumlahkan pada sum lokal (secara implisit) di suatu thread, kemudian akan dilakukan atomicAdd diakhir untuk memperoleh sum total global.
