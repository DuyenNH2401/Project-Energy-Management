import xml.etree.ElementTree as ET
import random

tree = ET.parse('cologne8_final.rou.xml')
root = tree.getroot()

# Duyệt qua từng flow trong file
for flow in root.findall('flow'):
    # Tạo thời gian bắt đầu ngẫu nhiên từ 0 đến 600 giây (10 phút đầu)
    new_begin = random.randint(0, 600) 
    # Thời gian kết thúc (ví dụ kéo dài thêm 3000 giây từ lúc bắt đầu)
    new_end = new_begin + 3000 
    
    # Cập nhật vào thuộc tính của flow
    flow.set('begin', str(new_begin))
    flow.set('end', str(new_end))

# Lưu lại file đã chỉnh sửa
tree.write('cologne8_final_random.rou.xml')
print("Đã xáo trộn thời gian khởi hành của các luồng xe!")