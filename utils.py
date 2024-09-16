import string
from datetime import datetime, timedelta
from io import StringIO
import csv
import re
import pandas as pd
import instaloader
import os
from dotenv import load_dotenv
from nltk.corpus import stopwords
import re
import string
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import emoji


def format_tanggal(bulan):
    now = datetime.now()
    return now - timedelta(days=bulan * 30)


def scraping(post_url, jumlah_komentar):
    scraped_data = []
    # Buat instance Instaloader
    L = instaloader.Instaloader()

     # Login ke Instagram
    USERNAME = ''
    PASSWORD = ''
    L.login(USERNAME, PASSWORD)

    # Dapatkan shortcode dari URL postingan
    shortcode = post_url.split("/")[-2]
    post = instaloader.Post.from_shortcode(L.context, shortcode)

    # Dapatkan komentar dan balasannya dari postingan
    comments = []
    usernames = []
    
    # Counter untuk membatasi jumlah komentar yang diambil
    counter = 0

    for comment in post.get_comments():
        if counter >= jumlah_komentar:
            break

        # Simpan komentar utama
        usernames.append(comment.owner.username)
        comments.append(comment.text)
        counter += 1
        
        # Periksa jika ada balasan (threaded comments)
        for reply in comment.answers:
            if counter >= jumlah_komentar:
                break

            # Gabungkan balasan dengan komentar
            usernames.append(reply.owner.username)
            comments.append(reply.text)
            counter += 1

    # Simpan data ke dalam DataFrame pandas
    scraped_data = pd.DataFrame({
        'username': usernames,
        'comments': comments
    })

    return scraped_data

# def scraping(post_url, jumlah_komentar):
#     scraped_data = []
#     L = instaloader.Instaloader()

#     USERNAME = 'suwandiaminsangaji'
    
#     # Coba untuk memuat sesi login dari file jika ada
#     try:
#         L.load_session_from_file(USERNAME)
#         print("Sesi login berhasil dimuat.")
#     except FileNotFoundError:
#         print("Tidak ada sesi login yang tersimpan, login menggunakan username dan password.")
#         PASSWORD = 'W@ndy110494;'
#         try:
#             L.login(USERNAME, PASSWORD)
#             # Simpan sesi login setelah berhasil login
#             L.save_session_to_file()
#             print("Login berhasil dan sesi telah disimpan.")
#         except instaloader.exceptions.LoginException as e:
#             print(f"Login gagal: {e}")
#             return None

#     # Dapatkan shortcode dari URL postingan
#     try:
#         shortcode = post_url.split("/")[-2]
#         post = instaloader.Post.from_shortcode(L.context, shortcode)
#     except Exception as e:
#         print(f"Error mendapatkan post dari shortcode: {e}")
#         return None

#     # Dapatkan komentar dan balasannya dari postingan
#     comments = []
#     usernames = []
    
#     # Counter untuk membatasi jumlah komentar yang diambil
#     counter = 0

#     try:
#         for comment in post.get_comments():
#             if counter >= jumlah_komentar:
#                 break

#             # Simpan komentar utama
#             usernames.append(comment.owner.username)
#             comments.append(comment.text)
#             counter += 1
            
#             # Periksa jika ada balasan (threaded comments)
#             for reply in comment.answers:
#                 if counter >= jumlah_komentar:
#                     break

#                 # Gabungkan balasan dengan komentar
#                 usernames.append(reply.owner.username)
#                 comments.append(reply.text)
#                 counter += 1
#     except Exception as e:
#         print(f"Error mengambil komentar: {e}")
#         return None

#     # Simpan data ke dalam DataFrame pandas
#     scraped_data = pd.DataFrame({
#         'username': usernames,
#         'comments': comments
#     })

#     return scraped_data


def generate_csv(scraped_data):
    csv_string = StringIO()

    if not scraped_data.empty:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_string)

        # Write header
        header = ['username', 'comments']
        csv_writer.writerow(header)

        # Write data rows
        for _, row in scraped_data.iterrows():
            csv_writer.writerow([row['username'], row['comments']])
    else:
        csv_writer.writerow(['username', 'comments'])  # Header only if no data

    csv_content = csv_string.getvalue()
    csv_string.close()

    return csv_content



def generate_csv_processing(data):
    csv_string = StringIO()  # Use StringIO as a file-like object

    if data:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_string)

        header = ['comments','text_tokenize', 'text_clean', 'label', 'score']
        csv_writer.writerow(header)

        # Write data rows
        for item in data:
            csv_writer.writerow([item['comments'], item['text_tokenize'], item['text_clean'], item['label'], item['score']])

    csv_content_processing = csv_string.getvalue()
    csv_string.close()

    return csv_content_processing

# Fungsi untuk tahap Cleansing
def preprocessing(text):
    # 1. Cleansing
    text = clean_text(text)

    # 2. Stopword Removal, Case Folding, Tokenizing
    stop_words = set(stopwords.words('indonesian'))
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # 3. Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words



def clean_text(text):
    if isinstance(text, str):
        # Remove usernames (pattern @username)
        text = re.sub(r'@\w+', '', text)
        
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'[‘’“”…]', '', text)
        text = re.sub(r'\n', ' ', text)
        
        # Hapus emoji
        text = emoji.replace_emoji(text, replace='')  # Menghapus semua emoji
        
        return text.strip()  # Menghapus spasi ekstra setelah semua pembersihan
    else:
        return ""

def remove_empty_rows(df, column_name):
    df_cleaned = df[df[column_name].apply(lambda x: len(clean_text(x)) > 0)]
    return df_cleaned


class Labelling:
    def __init__(self, dataset, lexicon_df):
        self.dataset = dataset
        self.lexicon_df = lexicon_df

    def labelling_data(self):
        df = pd.DataFrame(self.dataset)
        df['label'], df['score'] = zip(*[self.label_lexicon(df['comments'][i]) for i in range(len(df))])
        return df

    def label_lexicon(self, text):
        words = text.split()
        labels = []

        for word in words:
            label = self.lexicon_df.loc[self.lexicon_df['word'] == word, 'weight'].values
            if len(label) > 0:
                labels.append(label[0])

        sentiment_score = sum(labels)
        if sentiment_score > 0:
            return 'Positive', sentiment_score
        elif sentiment_score < 0:
            return 'Negative', sentiment_score
        else:
            return 'Netral', sentiment_score
        

def export(usernames, comments):
    fname = 'comments.xlsx'
    temp = {}
    temp.update({'username': usernames, 'comments': comments})
    df = pd.DataFrame(temp)
    df.to_excel(fname)

    print("Berhasil scraping data komentar")


