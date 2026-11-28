"""R03 analysis with explicit source provenance and pending LSTM verification."""
from pathlib import Path
import json
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
source=json.loads((ROOT/'references/R03/source_manifest.json').read_text())
check=json.loads((ROOT/'reports/R03_shape_check.json').read_text())
doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Cm(21),Cm(29.7)
sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Cm(2.54)
for name,size in [('Normal',11),('Title',18),('Heading 1',14),('Heading 2',12),('Caption',10)]:
 st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
 fonts=st.element.get_or_add_rPr().get_or_add_rFonts()
 for k in list(fonts.attrib):
  if 'theme' in k.lower():del fonts.attrib[k]
 for k in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+k),'Times New Roman')
for st in doc.styles:
 for b in list(st.element.iter(qn('w:pBdr'))):b.getparent().remove(b)
doc.styles['Normal'].paragraph_format.line_spacing=1.15
doc.styles['Normal'].paragraph_format.space_after=Pt(6)
doc.core_properties.author='Nguyễn Xuân Anh'
doc.core_properties.title='R03 Chuyển đổi LSTM và SegNet'
def p(t,bold=False):
 q=doc.add_paragraph();q.add_run(t).bold=bold;return q
def h(t):doc.add_heading(t,1)
def page():doc.add_page_break()
def pic(name,w=15.7):
 doc.add_picture(str(ROOT/'reports/R03_figures'/name),width=Cm(w));doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
def cap(t):
 q=doc.add_paragraph(t,style='Caption');q.alignment=WD_ALIGN_PARAGRAPH.CENTER
def table(headers,rows,widths):
 t=doc.add_table(rows=1,cols=len(headers));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
 for c,w in zip(t.columns,widths):c.width=Cm(w)
 for r,values in [(t.rows[0],headers)]+[(t.add_row(),x) for x in rows]:
  for i,value in enumerate(values):
   c=r.cells[i];c.width=Cm(widths[i]);c.text=str(value);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    el=OxmlElement('w:'+edge)
    for k,v in [('val','single'),('sz','4'),('color','D9D9D9')]:el.set(qn('w:'+k),v)
    b.append(el)
   pr.append(b)
   for para in c.paragraphs:
    para.paragraph_format.space_before=Pt(4);para.paragraph_format.space_after=Pt(4)
    if i>0:para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for run in para.runs:run.font.size=Pt(10)
 for c in t.rows[0].cells:
  el=OxmlElement('w:shd');el.set(qn('w:fill'),'D9EAF7');c._tc.get_or_add_tcPr().append(el)
  for r in c.paragraphs[0].runs:r.bold=True
 el=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(el);p('')

doc.add_paragraph('MỤC 3: CHI TIẾT CHUYỂN ĐỔI MODULE 3 (LSTM) VÀ MODULE 4 (SEGNET)',style='Title')
p('R03  |  Nguyễn Xuân Anh  |  MSSV: ................................',True)
p('Phạm vi: phân tích Module 3 CNN kết hợp LSTM và Module 4 phân đoạn bàn tay, kèm sơ đồ tensor và quy tắc dữ liệu. Việc huấn luyện hai module thuộc Quốc Việt và Tuyết Nhi; phần này không nhận là đã chạy lại các thí nghiệm của hai thành viên.')
h('3.1. Phân tích Chuyển đổi Module 3 - TimeDistributed CNN + LSTM')
p('Mỗi mẫu gồm một chuỗi ảnh chữ số MNIST; mục tiêu là dự đoán tổng các chữ số bằng một số thực. Theo task nhóm, chuỗi có 7 vị trí, input PyTorch [B,7,1,28,28] và target [B,1] dạng float32. Tổng có miền 0..63 khi mỗi vị trí là chữ số 0..9. Đây là hồi quy tổng số, không phải phân loại một chữ số hay 64 lớp tổng.')
p('CNN dùng chung trọng số để trích đặc trưng cho từng ảnh. LSTM nhận chuỗi đặc trưng và tổng hợp thông tin theo thời gian. Notebook Keras nguồn có ba khối Conv2D 64/128/256 kênh, mỗi khối dùng kernel 3, padding same, ReLU, MaxPooling 2 và Dropout 0,2; tiếp theo là hai LSTM 128 và Dropout 0,22, cuối cùng Dense một đầu ra tuyến tính.')
table(['Bước','Tensor với T=7','Ý nghĩa'],[
 ('Chuỗi đầu vào','[B,7,1,28,28]','7 ảnh grayscale'),('Gộp B và T','[B×7,1,28,28]','Cùng CNN cho mọi ảnh'),
 ('Ba khối CNN','[B×7,256,3,3]','28 → 14 → 7 → 3'),('Flatten và khôi phục T','[B,7,2304]','256×3×3 đặc trưng'),
 ('LSTM thứ nhất','[B,7,128]','Giữ output mọi bước'),('LSTM thứ hai','[B,7,128] → [B,128]','Lấy bước cuối'),('Linear cuối','[B,1]','Ước lượng tổng')],[4.2,5.5,6.2])
p('Luồng trên trình bày phương án chuyển đổi từ notebook Keras sang PyTorch theo yêu cầu chuỗi 7 ảnh trong đề cương nhóm. Phần LSTM là phân tích thiết kế từ mã nguồn Keras, không phải mô tả kết quả huấn luyện đã được tái chạy. Bản Keras gốc dùng tối đa 8 vị trí; khi áp dụng cấu hình 7 ảnh, Dataset, shape input và cách lấy tổng phải được đổi đồng bộ.')

page();doc.add_heading('3.1.1. TimeDistributed và hàm mất mát L1',2)
pic('lstm_tensor_flow.png');cap('Hình 3 1  Luồng tensor LSTM theo task 7 ảnh và CNN của notebook nguồn')
p('Trong Keras, TimeDistributed áp cùng một lớp lên từng bước thời gian. Với PyTorch có thể thực hiện tương đương bằng reshape: gộp B và T, chạy CNN một lần trên B×T ảnh, rồi trả tensor về [B,T,F]. Cách này dùng cùng trọng số CNN cho mọi ảnh và không làm trộn thứ tự chuỗi nếu các thao tác reshape giữ nguyên thứ tự mẫu và thời gian.')
for line in ['B, T, C, H, W = images.shape','features = cnn(images.reshape(B*T, C, H, W))','sequence = features.flatten(1).reshape(B, T, 2304)','out1, _ = lstm1(sequence)  # batch_first=True','out2, _ = lstm2(out1)','predicted_sum = head(dropout(out2[:, -1, :]))']:
 q=p(line)
 for run in q.runs:run.font.name='Consolas';run.font.size=Pt(9)
p('Đoạn trên minh họa chuyển đổi, không phải trích từ code PyTorch của Quốc Việt. Với hai LSTM, lớp đầu nhận input_size=2304 và hidden_size=128; lớp sau nhận input_size=128 và hidden_size=128. Tách hai nn.LSTM giúp thể hiện rõ hai giá trị return_sequences của Keras. nn.LSTM trả cả chuỗi output và trạng thái (h_n,c_n); batch_first=True chỉ thay bố trí input/output, không thay bố trí hidden state.')
p('nn.L1Loss với reduction="mean" tính MAE = trung bình |ŷ−y|. Đầu ra và target cùng [B,1], target là số thực; không áp Softmax hoặc làm tròn trước loss. Khi trình diễn có thể làm tròn sau suy luận nhưng vẫn phải báo cáo MAE trên số thực. Chênh lệch đơn vị của MAE là đơn vị tổng chữ số.')
table(['Nội dung','Notebook Keras nguồn','Cấu hình theo đề cương'],[
 ('Độ dài','maxToAdd=8; ngẫu nhiên 1..8 ảnh, đệm 0','7 vị trí; Dataset và model cần đồng bộ quy tắc đệm'),
 ('SGD và epoch','lr=0,01; 30 epoch','lr=0,01; 30 epoch'),('Batch size','128','64'),('Số đo','mean_absolute_error','L1Loss tương ứng MAE')],[3.5,6.2,6.2])

page();h('3.2. Phân tích Chuyển đổi Module 4 - Hand Depth SegNet')
p('Module 4 nhận ảnh depth grayscale 96×96 và dự đoán class index cho từng pixel thuộc 7 nhãn ngữ nghĩa. Đây là phân loại từng pixel: logits [B,7,96,96], mask đích [B,96,96]. Tên và ý nghĩa giải phẫu của 7 lớp phải lấy từ bộ dữ liệu, không tự suy ra chỉ từ giá trị mức xám.')
p(f"Phân tích này đối chiếu trực tiếp HandSegNet trong notebook 04_segmentation.ipynb của nhánh Nhi, commit {source['segnet_commit'][:12]}. Bản nhóm có ba encoder và ba decoder. Đã trích nguyên class và kiểm tra forward trên tensor tổng hợp để xác nhận kích thước cùng {check['params']:,} tham số; đây không phải chạy lại huấn luyện hoặc đánh giá trên ảnh thật.")
table(['Khối','Các lớp chính','Tensor đầu ra'],[
 ('Input','Ảnh depth một kênh','[B,1,96,96]'),
 ('Encoder 1','Conv 1→64→64, BN/ReLU, Pool 2','[B,64,48,48]'),
 ('Encoder 2','Conv 64→128→128, BN/ReLU, Pool 2','[B,128,24,24]'),
 ('Encoder 3','Conv 128→256→256, BN/ReLU, Pool 2','[B,256,12,12]'),
 ('Decoder 3','Upsample 2, Conv 256→256→128, BN/ReLU','[B,128,24,24]'),
 ('Decoder 2','Upsample 2, Conv 128→128→64, BN/ReLU','[B,64,48,48]'),
 ('Decoder 1','Upsample 2, Conv 64→64, BN/ReLU, Conv1×1 64→7','[B,7,96,96]')],[3,7.3,5.6])
p('Các Conv 3×3 dùng padding=1 để giữ H/W trước pooling. MaxPool2d(2,2) giảm mỗi chiều một nửa; ba lần pooling đưa 96 về 12. Decoder dùng nn.Upsample(scale_factor=2, mode="nearest") trước các convolution để tăng kích thước lần lượt 12→24→48→96. Conv cuối 1×1 chuyển 64 kênh đặc trưng thành 7 logits, không thêm Softmax trước CrossEntropyLoss.')
p('Bản triển khai của nhóm là encoder–decoder dùng nearest upsampling và không nối skip giữa encoder/decoder. Không mô tả nó như phép unpool khôi phục theo chỉ số MaxPool, vì notebook hiện tại không lưu hoặc dùng pooling indices. BatchNorm dùng thống kê theo batch khi train và thống kê đã lưu khi eval; do đó cần chuyển model.eval() trước suy luận.')

page();h('3.3. Sơ đồ Biến đổi Chiều Tensor')
pic('segnet_tensor_flow.png');cap('Hình 3 2  Luồng tensor được kiểm tra từ class SegNet của nhóm')
p('Giá trị B trong sơ đồ là batch thực tế; batch cuối có thể nhỏ hơn 8. Kiểm tra tổng hợp dùng B=2 và xác nhận đủ sáu khối trung gian. Bước argmax(dim=1) loại chiều lớp, tạo nhãn dự đoán [B,96,96]. Khi cần xác suất để quan sát mới áp Softmax trên chiều lớp, không áp vào input của loss.')
table(['Thao tác','Kích thước không gian','Kênh đầu ra'],[
 ('Input','96 × 96','1'),('Encoder 1 và pooling','48 × 48','64'),
 ('Encoder 2 và pooling','24 × 24','128'),('Encoder 3 và pooling','12 × 12','256'),
 ('Decoder 3','24 × 24','128'),('Decoder 2','48 × 48','64'),
 ('Decoder 1 logits','96 × 96','7'),('argmax theo lớp','96 × 96','Không còn chiều kênh')],[6.7,5,4.2])
p('Với Conv2d kernel 3, stride 1 và padding 1, mỗi chiều H/W được giữ nguyên. MaxPool2d kernel 2, stride 2 chia đôi kích thước; Upsample hệ số 2 nhân đôi kích thước. Vì 96 chia hết cho 2³, ba tầng nén và ba tầng giải nén khôi phục đúng 96×96 mà không cần crop hoặc bù pixel. Chiều kênh do từng Conv2d quyết định, không do Upsample thay đổi.')
p('Sơ đồ được đối chiếu bằng hook vào enc1, enc2, enc3, dec3, dec2 và dec1 của class HandSegNet. Kiểm tra tổng hợp xác nhận đúng các shape và loss chấp nhận cặp logits/mask. Không dùng tensor giả này để báo cáo accuracy, mIoU hoặc chất lượng mô hình đã huấn luyện.')
page();h('3.4. Thiết kế DataLoader và Tiêu chuẩn Mất mát SegNet')
p('Dataset ghép ảnh depth với mask tương ứng theo tên file: thay _depth_ bằng _labels_. Ảnh depth chuyển grayscale, resize 96×96 bằng bilinear rồi chia 255 thành float32 [1,96,96]. Mask resize bằng NEAREST để không tạo nhãn trung gian. Nếu mask nhiều kênh, pipeline hiện lấy kênh đầu; giả định này cần đúng với định dạng mask nguồn.')
table(['Pixel gốc','0','43','85','128','170','212','255'],[
 ('Class index','0','1','2','3','4','5','6')],[3.3,1.8,1.8,1.8,1.8,1.8,1.8,1.8])
p('Bảng tra LUT ánh xạ 7 mức pixel thành 0..6 và chuyển mask sang torch.long. make_hand_depth_loaders dùng batch_size=8: ảnh [B,1,96,96], mask [B,96,96]; shuffle cho train, giữ thứ tự test. Cần kiểm tra np.unique(mask) trước ánh xạ: LUT hiện khởi tạo bằng 0 nên pixel ngoài bảy mức có thể bị quy về nền nếu không được phát hiện.')
p('CrossEntropyLoss nhận logits [B,C,H,W] và target class index [B,H,W]. Về mặt toán học, loss lấy trung bình âm log xác suất lớp đúng trên pixel. Mask one-hot [B,7,H,W], mask RGB hoặc mask còn giá trị 255 không phải target index đúng cho pipeline này. Adam lr=0,0001, 15 epoch và batch 8 là cấu hình thấy trong notebook của nhóm.')

p('Khi kiểm tra dữ liệu, cần xác nhận số cặp ảnh–mask, giá trị mask nằm trong 0..6 sau LUT, dtype là long và không có augmentation ngẫu nhiên ở tập test. Tại vòng lặp train dùng model.train(), zero_grad(), backward() và step(); khi đánh giá chuyển model.eval() cùng no_grad(). Cách làm này giữ dropout/BatchNorm và gradient ở đúng chế độ.')
h('Tài liệu và tệp đối chiếu')
for text in [
 '[1] Bảng phân công và nhật ký thực nghiệm nhóm do người dùng cung cấp; yêu cầu T=7 và EXP_06 SGD 0,01, batch64, 30 epoch.',
 '[2] MNIST_CNN_LSTM.ipynb trong examples.zip, bản Keras nguồn; lưu tại references/R03/MNIST_CNN_LSTM_keras.ipynb.',
 '[3] Nhánh Nhi, notebook 04_segmentation.ipynb và src/datasets.py. Commit và SHA256 trong references/R03/source_manifest.json.',
 '[4] PyTorch LSTM: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html',
 '[5] PyTorch L1Loss và Upsample: https://docs.pytorch.org/docs/2.14/generated/torch.nn.L1Loss.html và https://docs.pytorch.org/docs/2.14/generated/torch.nn.Upsample.html',
 '[6] PyTorch CrossEntropyLoss: https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html']:
 q=p(text)
 for run in q.runs:run.font.size=Pt(9)
out=ROOT/'reports/R03_LSTM_SegNet_XuanAnh.docx';doc.save(out);print(out)
