from pypdf import PdfReader, PdfWriter
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io

# Input and output paths
pdf_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Way_waveform.pdf"
output_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Way_waveform_bw.pdf"

# Read the PDF
reader = PdfReader(pdf_path)
writer = PdfWriter()

# Process each page
for page_num in range(len(reader.pages)):
    page = reader.pages[page_num]

    # Get page dimensions
    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)

    # Create a new PDF page with same dimensions
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))

    found_image = False

    # Extract images from page
    if '/Resources' in page and '/XObject' in page['/Resources']:
        xObject = page['/Resources']['/XObject'].get_object()

        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                found_image = True
                # Get image data
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].get_data()

                # Create PIL Image
                try:
                    if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                        img = Image.frombytes('RGB', size, data)
                    elif xObject[obj]['/ColorSpace'] == '/DeviceGray':
                        img = Image.frombytes('L', size, data)
                    else:
                        # Try RGB as default
                        img = Image.frombytes('RGB', size, data)

                    # Convert to grayscale
                    img_gray = img.convert('L')

                    # Save to temporary buffer
                    img_buffer = io.BytesIO()
                    img_gray.save(img_buffer, format='PNG')
                    img_buffer.seek(0)

                    # Draw on canvas
                    can.drawImage(ImageReader(img_buffer), 0, 0, width=page_width, height=page_height)
                except Exception as e:
                    print(f"Error processing image on page {page_num}: {e}")
                    # If error, just copy the original page
                    found_image = False
                    break

    if found_image:
        can.save()
        packet.seek(0)
        # Add to writer
        new_reader = PdfReader(packet)
        if len(new_reader.pages) > 0:
            writer.add_page(new_reader.pages[0])
        else:
            writer.add_page(page)
    else:
        # No images or error, just copy the page
        writer.add_page(page)

# Write output
with open(output_path, 'wb') as output_file:
    writer.write(output_file)

print(f"已成功转换为黑白PDF: {output_path}")
