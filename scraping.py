import instaloader
import pandas as pd
import os
from instaloader.exceptions import QueryReturnedBadRequestException, ConnectionException, BadResponseException
import time

# Buat instance Instaloader
L = instaloader.Instaloader()

# Menambahkan User-Agent agar request tampak seperti dari browser
L.context.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

USERNAME = 'suwandiaminsangaji'
PASSWORD = 'W@n&y110494'

# Fungsi untuk login menggunakan sesi yang tersimpan atau login baru
def login_instagram():
    try:
        # Coba muat sesi dari file
        L.load_session_from_file(USERNAME)
        print(f"Berhasil memuat sesi yang tersimpan untuk {USERNAME}")
    except FileNotFoundError:
        print(f"Sesi tidak ditemukan, login dengan username dan password.")
        L.login(USERNAME, PASSWORD)
        L.save_session_to_file()

# Coba login atau muat sesi
login_instagram()

# Pilih postingan Instagram berdasarkan URL atau ID
POST_URL = 'https://www.instagram.com/p/C_xo3pbybKh/'

# Fungsi untuk mendapatkan data postingan dan komentar
def scrape_post_comments(post_url, retry_attempts=3, delay=60):
    attempt = 0
    while attempt < retry_attempts:
        try:
            # Mengambil shortcode dari URL
            post_shortcode = post_url.split("/")[-2]
            post = instaloader.Post.from_shortcode(L.context, post_shortcode)

            # Dapatkan komentar dan balasannya dari postingan
            comments = []
            usernames = []

            for comment in post.get_comments():
                # Simpan komentar utama
                usernames.append(comment.owner.username)
                comments.append(comment.text)

                # Periksa jika ada balasan (threaded comments)
                for reply in comment.answers:
                    # Gabungkan balasan dengan komentar
                    usernames.append(reply.owner.username)
                    comments.append(reply.text)

            # Simpan data ke dalam DataFrame pandas
            df = pd.DataFrame({
                'username': usernames,
                'komentar': comments
            })

            # Buat direktori jika belum ada
            output_directory = 'hasil_scraping'
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Simpan DataFrame ke file Excel
            output_file = os.path.join(output_directory, 'data.xlsx')
            df.to_excel(output_file, index=False)

            print("Scraping selesai. Data disimpan ke", output_file)
            break  # Hentikan loop jika berhasil

        except QueryReturnedBadRequestException as e:
            print("Instagram mengembalikan kesalahan permintaan (400 Bad Request):", e)
            print("Kemungkinan butuh CAPTCHA atau tindakan manual. Silakan tunggu atau coba nanti.")
            attempt += 1
            if attempt < retry_attempts:
                print(f"Menunggu {delay} detik sebelum mencoba lagi ({attempt}/{retry_attempts})...")
                time.sleep(delay)
            else:
                print("Gagal setelah beberapa kali percobaan.")
                break

        except ConnectionException as e:
            print("Terjadi kesalahan koneksi:", e)
            attempt += 1
            if attempt < retry_attempts:
                print(f"Menunggu {delay} detik sebelum mencoba lagi ({attempt}/{retry_attempts})...")
                time.sleep(delay)
            else:
                print("Gagal setelah beberapa kali percobaan.")
                break

        except BadResponseException as e:
            print("Respon buruk dari server:", e)
            attempt += 1
            if attempt < retry_attempts:
                print(f"Menunggu {delay} detik sebelum mencoba lagi ({attempt}/{retry_attempts})...")
                time.sleep(delay)
            else:
                print("Gagal setelah beberapa kali percobaan.")
                break

        except Exception as e:
            print("Terjadi kesalahan lain:", e)
            break

# Panggil fungsi untuk scrape komentar
scrape_post_comments(POST_URL)
