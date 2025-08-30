import os.path
import aspose.words as aw
import re
pdf_desc = r"D:\download\Описание ИТ.docx"
pdf_cs_path = r"D:\pycharm\vtb_hack_hh\module1\pdf_cv"
pdf_cv_list = [
    os.path.join(pdf_cs_path, file)
    for file in os.listdir(pdf_cs_path)
    if os.path.isfile(os.path.join(pdf_cs_path, file))
    ]

def clean_text(text: str) -> str:
    return re.sub(r'[\x00-\x1F\x7F]', '', text).strip()
def convert_to_text(file_list: list, file_num: int) -> str:
    """
    Function for converting file from list of links use in cycle to use on whole list.
    :param file_list: list with links to cv
    :param file_num: num of cv in list
    :return: text of file_num cv
    """
    lines_to_remove = [
        """Created with an evaluation copy of Aspose.Words. To remove all limitations, you can use Free Temporary 
        License  HYPERLINK "https://products.aspose.com/words/temporary-license/" 
        https://products.aspose.com/words/temporary-license/""",
        "Evaluation Only. Created with Aspose.Words. Copyright 2003-2025 Aspose Pty Ltd."
        ]
    file = file_list[file_num]
    _, ext = os.path.splitext(file)
    ext = ext.lower()
    if ext in [".doc", ".docx", ".rtf", ".pdf"]:
        try:
            doc = aw.Document(str(file))
            text = doc.get_text()
            text = text.splitlines()
            filtered_text = [line for line in text if line.strip() not in lines_to_remove]
            filtered_text = '\n'.join(filtered_text)
            return clean_text(filtered_text)
        except Exception as e:
            raise f"Ошибка при обработке {file}:{e}"
def convert_to_dict(file:str) -> dict:
    doc = aw.Document(file)
    tables = doc.get_child_nodes(aw.NodeType.TABLE,True)
    if not tables:
        raise ValueError(f"В {file} нет таблиц")
    table = tables[0]
    table = table.as_table()
    result = {}
    for row in table.rows:
        row = row.as_row()
        cell = row.cells
        key = clean_text(cell[0].get_text().strip())
        value = clean_text(cell[1].get_text().strip())
        if key:
            result[key] = value
    return result
