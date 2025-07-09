import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import processor

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Chatbot with Memory")
        self.root.geometry("900x650")
        self.chat_history = []

        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 12))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.configure(state='disabled')

        self.entry = tk.Entry(root, font=("Consolas", 12))
        self.entry.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=5)

        self.upload_button = tk.Button(root, text="📄 Upload PDF", command=self.upload_pdf)
        self.upload_button.pack(side=tk.LEFT, padx=5)

    def send_message(self, event=None):
        message = self.entry.get().strip()
        if not message:
            return
        self.display_message(f"✅ You:\n{message}\n\n")
        self.entry.delete(0, tk.END)
        self.chat_history.append({"role": "user", "content": message})
        threading.Thread(target=self.process_input, args=(message,)).start()

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            text = self.extract_pdf_text(file_path)
            if not text.strip():
                self.display_message("❌ Error: No text could be extracted from the PDF. Ensure Tesseract OCR is installed and in your PATH for image-based PDFs, or check if the PDF is corrupted.\n\n")
                return
            try:
                processor.load_pdf_into_memory(text)
                self.display_message("✅ PDF uploaded and indexed successfully.\n\n")
            except Exception as e:
                self.display_message(f"❌ Error indexing PDF: {e}\n\n")

    def extract_pdf_text(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            ocr_used = False
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                else:
                    try:
                        pix = page.get_pixmap(dpi=300)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img)
                        if ocr_text.strip():
                            text += ocr_text + "\n"
                            ocr_used = True
                        else:
                            print(f"OCR on page {page_num}: No text extracted.")
                    except Exception as ocr_error:
                        if "tesseract is not installed" in str(ocr_error).lower():
                            print(f"OCR error on page {page_num}: Tesseract is not installed or not in PATH. Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
                        else:
                            print(f"OCR error on page {page_num}: {ocr_error}")
                        continue
            print(f"Extracted text (first 500 chars): {text[:500]}...")
            if ocr_used:
                print("OCR was used for some pages.")
            with open("extracted_text.txt", "w", encoding="utf-8") as f:
                f.write(text)
            print("Extracted text saved to 'extracted_text.txt'.")
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""

    def process_input(self, input_data):
        response = processor.answer_question(input_data, self.chat_history)
        self.chat_history.append({"role": "assistant", "content": response})
        self.display_response(response)

    def display_message(self, message):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, message, "user")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def display_response(self, response):
        self.chat_area.configure(state='normal')
        lines = response.strip().split("\n")
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                self.chat_area.insert(tk.END, "\n🔧 Code:\n" if in_code else "\n", "code")
                continue
            tag = "code" if in_code else "ai"
            self.chat_area.insert(tk.END, line + "\n", tag)
        self.chat_area.insert(tk.END, "\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.tag_configure("user", foreground="green", font=("Consolas", 12))
        self.chat_area.tag_configure("ai", foreground="black", font=("Consolas", 12))
        self.chat_area.tag_configure("code", foreground="blue", font=("Courier New", 11))
        self.chat_area.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
