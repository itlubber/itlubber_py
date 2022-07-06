import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Border, Side, Font, PatternFill, Alignment
colors = ["2639E9", "F76E6C", "FE7715"]


def save_excel(df, writer, sheet_name, index=False):
    df.to_excel(writer, sheet_name=sheet_name, index=index)
    worksheet = writer.sheets[sheet_name]
    for i, col in enumerate(df.columns, 1):
        col_len = df[col].astype(str).apply(lambda x: len(x.encode("gbk"))).max()
        col_len = max(len(col.encode("gbk")), col_len) + 2
        worksheet.column_dimensions[get_column_letter(i)].width = col_len
        
        
def render_excel(excel_name, sheet_name=None, suffix="_已渲染"):
    """
    对 excel 文件进行格式渲染
    标题行填充颜色、文本白色楷体加粗
    文本楷体黑色、白色单元格填充
    最外层边框加粗、内层单元格边框常规、颜色为 2639E9
    """
    workbook = load_workbook(excel_name)
    
    if sheet_name and isinstance(worksheet, str):
        sheet_names = [sheet_name]
    else:
        sheet_names = workbook.get_sheet_names()
    
    for sheet_name in sheet_names:
        worksheet = workbook.get_sheet_by_name(sheet_name)
        
        sides = [
            Side(border_style="medium", color="2639E9"),
            Side(border_style="thin", color="2639E9"),
        ]
        
        for row_index, row in enumerate(worksheet.rows):
            if row_index == 0:
                for col_index, cell in enumerate(row):
                    cell.font = Font(size=10, name="楷体", color="FFFFFF", bold=True)
                    cell.fill = PatternFill(fill_type="solid", start_color="2639E9")
                    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=False)
                    
                    if col_index == 0:
                        cell.border = Border(left=sides[0], right=sides[1], top=sides[0], bottom=sides[0])
                    elif col_index == len(row) - 1:
                        cell.border = Border(left=sides[1], right=sides[0], top=sides[0], bottom=sides[0])
                    else:
                        cell.border = Border(left=sides[1], right=sides[1], top=sides[0], bottom=sides[0])
            else:
                for col_index, cell in enumerate(row):
                    cell.font = Font(size=10, name="楷体", color="000000")
                    cell.fill = PatternFill(fill_type="solid", start_color="FFFFFF")
                    cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
                    
                    if worksheet.max_row == 2:
                        if col_index == 0:
                            cell.border = Border(left=sides[0], right=sides[1], top=sides[0], bottom=sides[0])
                        elif col_index == len(row) - 1:
                            cell.border = Border(left=sides[1], right=sides[0], top=sides[0], bottom=sides[0])
                        else:
                            cell.border = Border(left=sides[1], right=sides[1], top=sides[0], bottom=sides[0])
                    else:
                        if row_index == 1:
                            if col_index == 0:
                                cell.border = Border(left=sides[0], right=sides[1], top=sides[0], bottom=sides[1])
                            elif col_index == len(row) - 1:
                                cell.border = Border(left=sides[1], right=sides[0], top=sides[0], bottom=sides[1])
                            else:
                                cell.border = Border(left=sides[1], right=sides[1], top=sides[0], bottom=sides[1])
                        elif row_index == worksheet.max_row - 1:
                            if col_index == 0:
                                cell.border = Border(left=sides[0], right=sides[1], top=sides[1], bottom=sides[0])
                            elif col_index == len(row) - 1:
                                cell.border = Border(left=sides[1], right=sides[0], top=sides[1], bottom=sides[0])
                            else:
                                cell.border = Border(left=sides[1], right=sides[1], top=sides[1], bottom=sides[0])
                        else:
                            if col_index == 0:
                                cell.border = Border(left=sides[0], right=sides[1], top=sides[1], bottom=sides[1])
                            elif col_index == len(row) - 1:
                                cell.border = Border(left=sides[1], right=sides[0], top=sides[1], bottom=sides[1])
                            else:
                                cell.border = Border(left=sides[1], right=sides[1], top=sides[1], bottom=sides[1])
                                
    excel_name_suffix = excel_name.split(".")[-1]
    workbook.save(excel_name.replace(excel_name_suffix, f"{suffix}{excel_name_suffix}"))
    workbook.close()
    
    
def excel_writer(excel_name, dataframe, sheet_name="Sheet1", index=False, suffix=""):
    with pd.ExcelWriter(excel_name, engine="openpyxl") as writer:
        save_excel(dataframe, writer, sheet_name, index=index)
    render_excel(excel_name, suffix=suffix)
    
    
if __name__ == "__main__":
    excel_writer("变量清单明细.xlsx", filter_info, sheet_name="附1_变量清单明细")
