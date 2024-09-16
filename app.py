from flask import Flask, render_template, request, Response, url_for, redirect
import pandas as pd
import os
from utils import preprocessing, Labelling, clean_text, remove_empty_rows, generate_csv, generate_csv_processing, scraping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from nbsvm import NBSVM
import io
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Global variables
scraped_data = []
hasil_processing = []
vectorizer = TfidfVectorizer()
svm_model = None
nbsvm_model = None
df_labeled = pd.DataFrame()

# Default C parameters based on kernel choices
default_C_param = {
    'linear': 1.0,
    'poly': 1.0,
    'sigmoid': 1.0,
    'rbf': 1.0
}

@app.route("/")
def index():
    return render_template("index.html")

# @app.route('/submit', methods=['POST'])
# def submit_form():
#     global scraped_data 
#     url = request.form.get('url')
#     jumlah = int(request.form.get('Jumlah', 10))

#     scraped_data = scraping(url, jumlah)

#     return render_template('scraping.html', data=scraped_data.to_dict(orient='records'))

@app.route('/submit', methods=['POST'])
def submit_form():
    global scraped_data 
    url = request.form.get('url')
    jumlah = int(request.form.get('Jumlah', 50))

    # Panggil fungsi scraping
    scraped_data = scraping(url, jumlah)

    # Periksa apakah scraping berhasil (scraped_data tidak None)
    if scraped_data is None:
        return render_template('error.html', message="Data scraping gagal. Coba lagi.")
    
    # Jika scraping berhasil, tampilkan hasilnya
    return render_template('scraping.html', data=scraped_data.to_dict(orient='records'))


@app.route("/scraping", methods=["GET", "POST"])
def route_scraping():
    scraped_data = None
    
    if request.method == "POST":
        post_url = request.form.get("post_url")
        jumlah_komentar = int(request.form.get("jumlah_komentar"))

        # Call the scraping function
        scraped_data = scraping(post_url, jumlah_komentar)

        # Convert DataFrame to list of dicts for passing to template
        if not scraped_data.empty:
            data = scraped_data.to_dict(orient='records')
        else:
            data = None
    
    return render_template('scraping.html', data=scraped_data)

@app.route('/download', methods=['POST'])
def download():
    global scraped_data

    # Memeriksa apakah scraped_data ada dan tidak kosong
    if scraped_data is None or scraped_data.empty:
        return "Error: No scraped data available."

    csv_data = generate_csv(scraped_data)
    response = Response(
        csv_data,
        content_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=comment_data.csv'}
    )
    return response


@app.route("/processing")
def processing():
    return render_template('processingData.html', data=hasil_processing)

@app.route('/process', methods=['POST'])
def process():
    global hasil_processing
    if request.method == 'POST':
        df = pd.read_csv(request.files['file'])

        # Preprocess the text data
        df['text_clean'] = df['comments'].apply(clean_text)

        # Remove empty rows after text cleaning
        df = remove_empty_rows(df, 'text_clean')

        # Tokenization and further preprocessing
        df['text_tokenize'] = df['text_clean'].apply(preprocessing)

        # Labeling
        lexicon_df = pd.read_csv('static/lexicon-word-dataset.csv')
        labeller = Labelling(df.to_dict(orient='records'), lexicon_df)
        df_labeled = labeller.labelling_data()

        hasil_processing = df_labeled.to_dict(orient='records')

        return render_template('processingData.html', data=hasil_processing)

@app.route('/download_processing', methods=['POST'])
def download_processing():
    csv_data_processing = generate_csv_processing(hasil_processing)
    response = Response(
        csv_data_processing,
        content_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=Hasil_Processing.csv'}
    )
    return response

@app.route('/nbsvm', methods=['GET', 'POST'])
def nbsvm():
    global hasil_processing, nbsvm_model, df_labeled

    if request.method == 'POST':
        uploaded_file = request.files['file']
        # kernel_choice = request.form['svmKarnel'].strip()
        kernel_choice = request.form.get('svmKarnel', 'linear').strip()
        split_ratio = float(request.form['split_ratio'])

        if uploaded_file.filename != '':
            # Preprocessing data
            df = pd.read_csv(uploaded_file)
            df['text_tokenize'] = df['comments'].apply(preprocessing)
            df['comments'] = df['comments'].apply(clean_text)

            # Lexicon-based labeling
            lexicon_df = pd.read_csv('static/lexicon-word-dataset.csv')
            labeller = Labelling(df.to_dict(orient='records'), lexicon_df)
            df_labeled = labeller.labelling_data()

            # Check if enough classes exist for classification
            num_classes = len(df_labeled['label'].unique())
            if num_classes < 2:
                return "Insufficient number of classes for classification."

            # Split the data into training and test sets
            X = df_labeled['comments']
            y = df_labeled['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

            # Initialize and train NBSVM model
            nbsvm_model = NBSVM()

            valid_kernels = ['linear', 'poly', 'sigmoid', 'rbf']
            if kernel_choice not in valid_kernels:
                return "Invalid kernel selection"

            # Default C parameter values for different kernels
            default_C_param = {'linear': 1.0, 'poly': 0.5, 'rbf': 1.0, 'sigmoid': 1.0}
            C_param_value = default_C_param.get(kernel_choice, 1.0)

            # Fit the NBSVM model
            nbsvm_model.fit(X_train, y_train, C=C_param_value, kernel=kernel_choice)

            # Predict and evaluate the model
            y_pred = nbsvm_model.predict(X_test)
            hasil_processing = df_labeled.to_dict(orient='records')

            # Output evaluation metrics
            report = classification_report(y_test, y_pred, output_dict=True)

            # Redirect to the endpoint for displaying the results
            return redirect(url_for('hasil_sentimen_nbsvm'))

    return render_template('nbsvm.html')

# @app.route('/hasil_sentimen_nbsvm')
# def hasil_sentimen_nbsvm():
#     global hasil_processing, nbsvm_model, df_labeled

#     hasil_sentimen_nbsvm = []
#     actual_labels = []

#     for data in hasil_processing:
#         prediction = nbsvm_model.predict([data['comments']])[0]
#         hasil_sentimen_nbsvm.append({'comments': data['comments'], 'sentiment': prediction})
#         actual_labels.append(data['label'])

#     y_pred = [result['sentiment'] for result in hasil_sentimen_nbsvm]
#     y_true = actual_labels

#     cm = confusion_matrix(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')

#     all_text = ' '.join(df_labeled['comments'])
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
#     plt.figure(figsize=(18, 10))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')

#     image_dir = 'static/image/nbsvm'
#     os.makedirs(image_dir, exist_ok=True)
#     wordcloud_path = os.path.join(image_dir, 'wordcloud_nbsvm.png')
#     plt.savefig(wordcloud_path)
#     plt.close()

#     labels = ['Negative', 'Neutral', 'Positive']
#     num_classes = cm.shape[0]
#     sizes = [cm[i, i] if i < num_classes else 0 for i in range(len(labels))]
#     explode = (0.1, 0, 0)
#     colors = ['red', 'orange', 'green']

#     fig1, ax1 = plt.subplots()
#     ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
#     ax1.axis('equal')

#     pie_chart_path = os.path.join(image_dir, 'pie_chart_nbsvm.png')
#     plt.savefig(pie_chart_path)
#     plt.close('all')

#     report_nbsvm = classification_report(y_true, y_pred)
#     plt.figure(figsize=(10, 6))
#     plt.text(0.5, 0.5, report_nbsvm, ha='center', va='center', fontsize=16)
#     plt.axis('off')

#     img_buffer = io.BytesIO()
#     plt.savefig(img_buffer, format='png', bbox_inches='tight')
#     img_buffer.seek(0)
#     img_str = base64.b64encode(img_buffer.read()).decode()
#     plt.close()

#     img_dir = 'static/image/report/nbsvm'
#     os.makedirs(img_dir, exist_ok=True)
#     img_path = os.path.join(img_dir, 'classification_report_nbsvm.png')
#     with open(img_path, 'wb') as img_file:
#         img_file.write(base64.b64decode(img_str))

#     return render_template('nbsvm.html', hasil_sentimen_nbsvm=hasil_sentimen_nbsvm, confusion_matrix=cm, accuracy=accuracy, precision=precision, recall=recall, f1=f1, wordcloud_path=wordcloud_path, pie_chart_path=pie_chart_path)

@app.route('/hasil_sentimen_nbsvm')
def hasil_sentimen_nbsvm():
    global hasil_processing, nbsvm_model, df_labeled

    hasil_sentimen_nbsvm = []
    actual_labels = []

    for data in hasil_processing:
        prediction = nbsvm_model.predict([data['comments']])[0]
        hasil_sentimen_nbsvm.append({'comments': data['comments'], 'sentiment': prediction})
        actual_labels.append(data['label'])

    y_pred = [result['sentiment'] for result in hasil_sentimen_nbsvm]
    y_true = actual_labels

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Generate Wordcloud and Pie Chart
    all_text = ' '.join(df_labeled['comments'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(18, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    image_dir = 'static/image/nbsvm'
    os.makedirs(image_dir, exist_ok=True)
    wordcloud_path = os.path.join(image_dir, 'wordcloud_nbsvm.png')
    plt.savefig(wordcloud_path)
    plt.close()

    labels = ['Negative', 'Neutral', 'Positive']
    num_classes = cm.shape[0]
    sizes = [cm[i, i] if i < num_classes else 0 for i in range(len(labels))]
    explode = (0.1, 0, 0)
    colors = ['red', 'orange', 'green']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
    ax1.axis('equal')

    pie_chart_path = os.path.join(image_dir, 'pie_chart_nbsvm.png')
    plt.savefig(pie_chart_path)
    plt.close('all')

    report_nbsvm = classification_report(y_true, y_pred)

    return render_template(
        'nbsvm.html', 
        hasil_sentimen_nbsvm=hasil_sentimen_nbsvm, 
        confusion_matrix=cm, 
        accuracy=accuracy, 
        precision=precision, 
        recall=recall, 
        f1=f1, 
        wordcloud_path=wordcloud_path, 
        pie_chart_path=pie_chart_path
    )




@app.route('/hasil')
def hasil():
    global nbsvm_model, hasil_processing

    print("Hasil processing: ", nbsvm_model)

    hasil_nbsvm = []
    cm_nbsvm = ''
    accuracy_nbsvm = ''
    precision_nbsvm = ''
    recall_nbsvm = ''
    f1_nbsvm = ''
    support_nbsvm = ''
    
    try:
        hasil_nbsvm = []
        actual_labels_nbsvm = []
        predicted_labels_nbsvm = []
        
        for data in hasil_processing:
            text_vectorized = vectorizer.transform([data['comments']])
            prediction = nbsvm_model.predict(text_vectorized)[0]
            hasil_nbsvm.append({'comments': data['comments'], 'sentiment': prediction})
            actual_labels_nbsvm.append(data['label'])
            predicted_labels_nbsvm.append(prediction)
        
        # Menghitung metrik evaluasi
        cm_nbsvm = confusion_matrix(actual_labels_nbsvm, predicted_labels_nbsvm)
        accuracy_nbsvm = accuracy_score(actual_labels_nbsvm, predicted_labels_nbsvm)
        precision_nbsvm = precision_score(actual_labels_nbsvm, predicted_labels_nbsvm, average='weighted', zero_division=0)
        recall_nbsvm = recall_score(actual_labels_nbsvm, predicted_labels_nbsvm, average='weighted', zero_division=0)
        f1_nbsvm = f1_score(actual_labels_nbsvm, predicted_labels_nbsvm, average='weighted', zero_division=0)
        report_nbsvm = classification_report(actual_labels_nbsvm, predicted_labels_nbsvm, zero_division=0)
        support_nbsvm = report_nbsvm.split("\n")[-2]  # Ambil baris dukungan
    except Exception as e:
        print(f"Error during nbsvm processing: {e}")
        cm_nbsvm = 'Error'
        accuracy_nbsvm = 'Error'
        precision_nbsvm = 'Error'
        recall_nbsvm = 'Error'
        f1_nbsvm = 'Error'
        support_nbsvm = 'Error'
    
    return render_template('hasil.html', 
                           hasil_nbsvm=hasil_nbsvm, cm_nbsvm=cm_nbsvm, accuracy_nbsvm=accuracy_nbsvm, 
                           precision_nbsvm=precision_nbsvm,
                           recall_nbsvm=recall_nbsvm,
                           f1_nbsvm=f1_nbsvm,
                           support_nbsvm=support_nbsvm)





if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

