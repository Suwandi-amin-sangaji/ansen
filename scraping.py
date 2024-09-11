import instaloader
import pandas as pd
import os
from dotenv import load_dotenv




# Buat instance Instaloader
L = instaloader.Instaloader()
load_dotenv()

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
L.login(USERNAME, PASSWORD)

# Pilih postingan Instagram berdasarkan URL atau ID
POST_URL = 'https://www.instagram.com/p/C_PGVSVy-mH/?img_index=1'
post = instaloader.Post.from_shortcode(L.context, POST_URL.split("/")[-2])

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
