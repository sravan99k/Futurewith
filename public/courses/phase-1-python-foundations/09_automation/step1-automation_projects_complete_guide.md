---
title: "Python Automation Projects: Practical Office & Web Automation"
level: "Intermediate"
time: "200 mins"
prereq: "python_fundamentals_complete_guide.md"
tags:
  ["python", "automation", "excel", "web-scraping", "api", "office-automation"]
---

# 🤖 Python Automation Projects: Automate Your Workflow

_Build Practical Automation Solutions for Real-World Tasks_

---

## 📘 **VERSION & UPDATE INFO**

**📘 Version 2.1 — Updated: November 2025**  
_Future-ready automation with modern tools and techniques_

**🟡 Intermediate**  
_Essential for productivity enhancement, business process automation, and data management_

**🏢 Used in:** Business automation, data processing, web scraping, API integration, report generation  
**🧰 Popular Tools:** openpyxl, pandas, requests, BeautifulSoup, Selenium, schedule, os, shutil

**🔗 Cross-reference:** Connect with `python_problem_solving_mindset_complete_guide.md` and `python_industry_applications_complete_guide.md`

---

**💼 Career Paths:** Automation Engineer, Business Analyst, Data Engineer, Operations Manager  
**🎯 Master Level:** Build comprehensive automation solutions for complex workflows

**🎯 Learning Navigation Guide**  
**If you score < 70%** → Start with simple Excel automation and basic web scraping  
**If you score ≥ 80%** → Build complex multi-step automation workflows and scheduled tasks

---

## 📊 **Excel Automation & Spreadsheet Processing**

### **Excel File Operations with openpyxl**

```python
import openpyxl
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from typing import List, Dict, Any, Optional
import datetime
from pathlib import Path

# 1. Excel File Manager Class
class ExcelAutomation:
    """Comprehensive Excel automation class"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.workbook = None
        self.worksheet = None

    def load_workbook(self, sheet_name: str = None) -> None:
        """Load Excel workbook"""
        try:
            self.workbook = load_workbook(self.file_path)
            if sheet_name:
                self.worksheet = self.workbook[sheet_name]
            else:
                self.worksheet = self.workbook.active
            print(f"📊 Loaded workbook: {self.file_path}")
        except Exception as e:
            print(f"❌ Error loading workbook: {e}")

    def create_workbook(self, sheet_name: str = "Sheet1") -> None:
        """Create new workbook"""
        self.workbook = Workbook()
        self.worksheet = self.workbook.active
        self.worksheet.title = sheet_name
        print(f"📄 Created new workbook with sheet: {sheet_name}")

    def read_data(self, start_row: int = 1, end_row: int = None,
                 start_col: int = 1, end_col: int = None) -> List[List]:
        """Read data from worksheet"""
        if not self.worksheet:
            raise ValueError("No worksheet loaded")

        data = []
        max_row = end_row or self.worksheet.max_row
        max_col = end_col or self.worksheet.max_column

        for row in range(start_row, max_row + 1):
            row_data = []
            for col in range(start_col, max_col + 1):
                cell_value = self.worksheet.cell(row=row, column=col).value
                row_data.append(cell_value)
            data.append(row_data)

        return data

    def write_data(self, data: List[List], start_row: int = 1, start_col: int = 1) -> None:
        """Write data to worksheet"""
        if not self.worksheet:
            raise ValueError("No worksheet loaded")

        for row_idx, row_data in enumerate(data):
            for col_idx, value in enumerate(row_data):
                cell = self.worksheet.cell(
                    row=start_row + row_idx,
                    column=start_col + col_idx
                )
                cell.value = value

    def add_headers(self, headers: List[str], row: int = 1, col: int = 1) -> None:
        """Add formatted headers"""
        for col_idx, header in enumerate(headers):
            cell = self.worksheet.cell(row=row, column=col + col_idx)
            cell.value = header

            # Format header
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")

    def format_cells(self, start_row: int, start_col: int,
                    end_row: int, end_col: int,
                    font_size: int = 11, bold: bool = False,
                    bg_color: str = None, text_color: str = None) -> None:
        """Format cell range"""
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                cell = self.worksheet.cell(row=row, column=col)

                if font_size:
                    cell.font = Font(size=font_size, bold=bold)
                if bg_color:
                    cell.fill = PatternFill(start_color=bg_color, end_color=bg_color, fill_type="solid")
                if text_color:
                    cell.font = Font(color=text_color)

    def add_borders(self, start_row: int, start_col: int,
                   end_row: int, end_col: int) -> None:
        """Add borders to cell range"""
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                cell = self.worksheet.cell(row=row, column=col)
                cell.border = thin_border

    def auto_fit_columns(self, start_col: int = 1, end_col: int = None) -> None:
        """Auto-fit column widths"""
        max_col = end_col or self.worksheet.max_column
        for col in range(start_col, max_col + 1):
            column_letter = get_column_letter(col)
            self.worksheet.column_dimensions[column_letter].width = 15

    def save_workbook(self, file_path: str = None) -> None:
        """Save workbook to file"""
        save_path = file_path or self.file_path
        self.workbook.save(save_path)
        print(f"💾 Saved workbook: {save_path}")

# 2. Data Processing with Pandas
class ExcelDataProcessor:
    """Advanced data processing with pandas"""

    def __init__(self):
        self.dataframes = {}

    def load_excel_to_dataframe(self, file_path: str, sheet_name: str = None) -> pd.DataFrame:
        """Load Excel file to pandas DataFrame"""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)

            print(f"📊 Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return pd.DataFrame()

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess DataFrame"""
        print("🧹 Cleaning data...")

        # Remove empty rows
        df = df.dropna(how='all')

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove empty columns
        df = df.dropna(axis=1, how='all')

        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

        # Convert data types
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass

        print(f"✅ Cleaned data: {len(df)} rows, {len(df.columns)} columns")
        return df

    def filter_data(self, df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
        """Filter DataFrame based on conditions"""
        filtered_df = df.copy()

        for column, condition in conditions.items():
            if column in df.columns:
                if isinstance(condition, dict):
                    # Range condition
                    if 'min' in condition:
                        filtered_df = filtered_df[filtered_df[column] >= condition['min']]
                    if 'max' in condition:
                        filtered_df = filtered_df[filtered_df[column] <= condition['max']]
                elif isinstance(condition, list):
                    # List of values
                    filtered_df = filtered_df[filtered_df[column].isin(condition)]
                else:
                    # Exact match
                    filtered_df = filtered_df[filtered_df[column] == condition]

        print(f"🔍 Filtered data: {len(filtered_df)} rows")
        return filtered_df

    def create_summary_report(self, df: pd.DataFrame,
                            group_by_column: str,
                            value_column: str) -> pd.DataFrame:
        """Create summary report"""
        summary = df.groupby(group_by_column)[value_column].agg([
            'count', 'sum', 'mean', 'min', 'max', 'std'
        ]).round(2)

        summary.columns = ['Count', 'Total', 'Average', 'Min', 'Max', 'Std_Dev']
        summary['Percentage'] = (summary['Total'] / summary['Total'].sum() * 100).round(1)

        print(f"📈 Created summary report by {group_by_column}")
        return summary

    def export_to_excel(self, dataframes: Dict[str, pd.DataFrame],
                       file_path: str) -> None:
        """Export multiple DataFrames to Excel"""
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"📄 Exported {sheet_name} with {len(df)} rows")

        print(f"💾 Exported to {file_path}")

# 3. Report Generator
class ExcelReportGenerator:
    """Generate professional Excel reports"""

    def __init__(self):
        self.workbook = Workbook()
        self.workbook.remove(self.workbook.active)  # Remove default sheet

    def create_sales_report(self, data: List[Dict],
                          output_file: str = "sales_report.xlsx") -> None:
        """Create sales performance report"""

        # Prepare data
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')

        # Summary by month
        monthly_summary = df.groupby('month').agg({
            'revenue': 'sum',
            'quantity': 'sum',
            'profit': 'sum'
        }).round(2)

        # Add to workbook
        self.add_data_sheet("Sales Data", df)
        self.add_summary_sheet("Monthly Summary", monthly_summary)
        self.add_charts_sheet(df, monthly_summary)

        # Save
        self.workbook.save(output_file)
        print(f"📊 Sales report created: {output_file}")

    def add_data_sheet(self, sheet_name: str, df: pd.DataFrame) -> None:
        """Add data sheet with formatting"""
        ws = self.workbook.create_sheet(title=sheet_name)

        # Add headers
        headers = list(df.columns)
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col)
            cell.value = header
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")

        # Add data
        for row_idx, (_, row) in enumerate(df.iterrows(), 2):
            for col_idx, value in enumerate(row, 1):
                ws.cell(row=row_idx, column=col_idx).value = value

        # Auto-fit columns
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 15

    def add_summary_sheet(self, sheet_name: str, summary_df: pd.DataFrame) -> None:
        """Add summary sheet"""
        ws = self.workbook.create_sheet(title=sheet_name)

        # Add title
        ws['A1'] = "Summary Report"
        ws['A1'].font = Font(size=16, bold=True)
        ws['A1'].fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        ws['A1'].font = Font(size=16, bold=True, color="FFFFFF")

        # Add data starting from row 3
        summary_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
        for row_idx, row_data in enumerate(summary_data, 3):
            for col_idx, value in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.value = value

                # Format header row
                if row_idx == 3:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")

        # Auto-fit columns
        for col in range(1, len(summary_df.columns) + 2):
            ws.column_dimensions[get_column_letter(col)].width = 12

    def add_charts_sheet(self, df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        """Add charts sheet"""
        ws = self.workbook.create_sheet(title="Charts")

        # Add chart title
        ws['A1'] = "Sales Performance Charts"
        ws['A1'].font = Font(size=14, bold=True)

        # Prepare data for charts
        chart_data = summary_df.reset_index()
        chart_data['month_str'] = chart_data['month'].astype(str)

        # Add chart data
        headers = ['Month', 'Revenue', 'Quantity', 'Profit']
        for col, header in enumerate(headers, 1):
            ws.cell(row=3, column=col).value = header
            ws.cell(row=3, column=col).font = Font(bold=True)

        for row_idx, row in chart_data.iterrows():
            ws.cell(row=row_idx + 4, column=1).value = row['month_str']
            ws.cell(row=row_idx + 4, column=2).value = row['revenue']
            ws.cell(row=row_idx + 4, column=3).value = row['quantity']
            ws.cell(row=row_idx + 4, column=4).value = row['profit']

# Demo Excel automation
def demo_excel_automation():
    """Demonstrate Excel automation capabilities"""
    print("📊 Excel Automation Demo")
    print("=" * 40)

    # Create sample data
    sample_data = [
        {"name": "John Doe", "age": 30, "salary": 50000, "department": "Sales"},
        {"name": "Jane Smith", "age": 25, "salary": 45000, "department": "Marketing"},
        {"name": "Bob Johnson", "age": 35, "salary": 60000, "department": "IT"},
        {"name": "Alice Brown", "age": 28, "salary": 52000, "department": "HR"},
        {"name": "Charlie Davis", "age": 32, "salary": 55000, "department": "Sales"}
    ]

    # Excel automation demo
    excel_auto = ExcelAutomation("demo.xlsx")
    excel_auto.create_workbook("Employee Data")

    # Add headers and data
    headers = ["Name", "Age", "Salary", "Department"]
    excel_auto.add_headers(headers)
    excel_auto.write_data(sample_data, start_row=2)

    # Format data
    excel_auto.format_cells(2, 1, 6, 4, font_size=11)
    excel_auto.add_borders(1, 1, 6, 4)
    excel_auto.auto_fit_columns()

    excel_auto.save_workbook("employee_data.xlsx")

    # Data processing demo
    processor = ExcelDataProcessor()
    df = pd.DataFrame(sample_data)
    print(f"📈 DataFrame shape: {df.shape}")

    # Clean data
    clean_df = processor.clean_dataframe(df)
    print(f"🧹 Cleaned shape: {clean_df.shape}")

    # Filter data
    filtered_df = processor.filter_data(clean_df, {"salary": {"min": 50000}})
    print(f"🔍 Filtered employees with salary >= 50000: {len(filtered_df)}")

    # Create summary
    summary = processor.create_summary_report(clean_df, "department", "salary")
    print("📊 Department Summary:")
    print(summary)

    # Generate sales report
    sales_data = [
        {"date": "2025-01-15", "product": "Widget A", "quantity": 100, "revenue": 5000, "profit": 1500},
        {"date": "2025-01-20", "product": "Widget B", "quantity": 75, "revenue": 3750, "profit": 1200},
        {"date": "2025-02-10", "product": "Widget A", "quantity": 120, "revenue": 6000, "profit": 1800},
        {"date": "2025-02-15", "product": "Widget C", "quantity": 50, "revenue": 2500, "profit": 800},
        {"date": "2025-03-05", "product": "Widget B", "quantity": 90, "revenue": 4500, "profit": 1400}
    ]

    report_gen = ExcelReportGenerator()
    report_gen.create_sales_report(sales_data, "sales_report_demo.xlsx")

    print("✅ Excel automation demo completed!")

# Run the demo
# demo_excel_automation()
```

### **Advanced Excel Features**

```python
# 4. Excel Formula Integration
class ExcelFormulaManager:
    """Manage Excel formulas and calculations"""

    def __init__(self, worksheet):
        self.ws = worksheet

    def add_sum_formula(self, start_cell: str, end_cell: str, result_cell: str) -> None:
        """Add SUM formula"""
        self.ws[result_cell] = f"=SUM({start_cell}:{end_cell})"

    def add_average_formula(self, start_cell: str, end_cell: str, result_cell: str) -> None:
        """Add AVERAGE formula"""
        self.ws[result_cell] = f"=AVERAGE({start_cell}:{end_cell})"

    def add_vlookup_formula(self, lookup_value: str, table_range: str,
                          col_index: int, result_cell: str) -> None:
        """Add VLOOKUP formula"""
        self.ws[result_cell] = f"=VLOOKUP({lookup_value},{table_range},{col_index},FALSE)"

    def add_conditional_formula(self, condition: str, true_value: str,
                              false_value: str, result_cell: str) -> None:
        """Add IF formula"""
        self.ws[result_cell] = f"=IF({condition},{true_value},{false_value})"

    def add_percentage_change(self, old_value_cell: str, new_value_cell: str,
                            result_cell: str) -> None:
        """Add percentage change formula"""
        self.ws[result_cell] = f"=({new_value_cell}-{old_value_cell})/{old_value_cell}"

    def create_sales_tax_formula(self, amount_cell: str, tax_rate: float,
                               result_cell: str) -> None:
        """Add sales tax calculation"""
        self.ws[result_cell] = f"={amount_cell}*{tax_rate}"

    def add_date_calculations(self, start_date_cell: str, end_date_cell: str,
                            days_cell: str, months_cell: str) -> None:
        """Add date calculations"""
        # Days between dates
        self.ws[days_cell] = f"={end_date_cell}-{start_date_cell}"
        # Months between dates (approximate)
        self.ws[months_cell] = f"=DATEDIF({start_date_cell},{end_date_cell},\"m\")"

# 5. Excel Pivot Table Creation
class ExcelPivotTable:
    """Create Excel pivot tables"""

    def __init__(self, source_sheet, pivot_sheet):
        self.source_sheet = source_sheet
        self.pivot_sheet = pivot_sheet

    def create_simple_pivot(self, data_range: str, row_field: str,
                          value_field: str, pivot_cell: str = "A1") -> None:
        """Create simple pivot table"""
        from openpyxl.worksheet.table import Table, TableStyleInfo

        # This is a simplified version - full pivot tables require complex setup
        # For production use, consider using pandas pivot_table() instead
        print("📊 Creating pivot table (simplified)...")

        # Get data for pivot
        data = self._extract_pivot_data(data_range)
        df = pd.DataFrame(data)

        # Create pivot summary
        pivot_summary = df.groupby(row_field)[value_field].sum().reset_index()

        # Write to pivot sheet
        self.pivot_sheet[pivot_cell] = f"Summary by {row_field}"
        for row_idx, (_, row) in enumerate(pivot_summary.iterrows(), 2):
            self.pivot_sheet[f"A{row_idx}"] = row[row_field]
            self.pivot_sheet[f"B{row_idx}"] = row[value_field]

    def _extract_pivot_data(self, data_range: str) -> List[Dict]:
        """Extract data for pivot table"""
        # Parse range (e.g., "A1:D10")
        start_cell, end_cell = data_range.split(":")

        # Get dimensions
        max_row = self.source_sheet.max_row
        max_col = self.source_sheet.max_column

        # Extract data
        data = []
        for row in range(2, max_row + 1):  # Skip header
            row_data = {}
            for col in range(1, max_col + 1):
                header = self.source_sheet.cell(row=1, column=col).value
                value = self.source_sheet.cell(row=row, column=col).value
                if header:
                    row_data[header] = value
            data.append(row_data)

        return data

# 6. Excel Charts and Graphs
class ExcelChartManager:
    """Create charts and graphs in Excel"""

    def __init__(self, worksheet):
        self.ws = worksheet

    def create_column_chart(self, title: str, x_axis: str, y_axis: str,
                          data_range: str, chart_position: str = "F2") -> None:
        """Create column chart"""
        from openpyxl.chart import BarChart, Reference

        # Create chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = title
        chart.y_axis.title = y_axis
        chart.x_axis.title = x_axis

        # Add data
        data = Reference(self.ws, min_col=2, min_row=1, max_row=10, max_col=3)
        cats = Reference(self.ws, min_col=1, min_row=2, max_row=10)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)

        # Place chart
        self.ws.add_chart(chart, chart_position)
        print(f"📊 Created column chart: {title}")

# Demo advanced Excel features
def demo_advanced_excel():
    """Demonstrate advanced Excel features"""
    print("📈 Advanced Excel Features Demo")
    print("=" * 40)

    # Create workbook with formulas
    wb = Workbook()
    ws = wb.active
    ws.title = "Formula Demo"

    # Add sample data
    ws['A1'] = "Item"
    ws['B1'] = "Price"
    ws['C1'] = "Quantity"
    ws['D1'] = "Total"
    ws['E1'] = "Tax (10%)"
    ws['F1'] = "Final Price"

    # Add data
    items = [["Apples", 2.5, 10], ["Bananas", 1.2, 15], ["Oranges", 3.0, 8]]
    for row_idx, item in enumerate(items, 2):
        ws[f'A{row_idx}'] = item[0]
        ws[f'B{row_idx}'] = item[1]
        ws[f'C{row_idx}'] = item[2]
        ws[f'D{row_idx}'] = f"=B{row_idx}*C{row_idx}"  # Total
        ws[f'E{row_idx}'] = f"=D{row_idx}*0.1"        # Tax
        ws[f'F{row_idx}'] = f"=D{row_idx}+E{row_idx}" # Final

    # Add totals
    ws['A6'] = "TOTALS:"
    ws['D6'] = "=SUM(D2:D4)"
    ws['E6'] = "=SUM(E2:E4)"
    ws['F6'] = "=SUM(F2:F4)"

    # Format totals
    for col in ['D', 'E', 'F']:
        ws[f'{col}6'].font = Font(bold=True)
        ws[f'{col}6'].fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    # Create formula manager
    formula_mgr = ExcelFormulaManager(ws)

    # Add additional formulas
    formula_mgr.add_sum_formula("D2", "D4", "D8")
    formula_mgr.add_average_formula("B2", "B4", "B8")
    formula_mgr.add_percentage_change("D2", "D4", "D9")

    # Save workbook
    wb.save("advanced_formulas.xlsx")
    print("💾 Created advanced Excel file: advanced_formulas.xlsx")

    # Create chart
    chart_ws = wb.create_sheet("Sales Chart")
    chart_data = [["Month", "Sales"], ["Jan", 1000], ["Feb", 1200], ["Mar", 1100]]

    for row_idx, data in enumerate(chart_data, 1):
        for col_idx, value in enumerate(data, 1):
            chart_ws.cell(row=row_idx, column=col_idx).value = value

    chart_mgr = ExcelChartManager(chart_ws)
    chart_mgr.create_column_chart("Monthly Sales", "Month", "Sales", "A1:B4", "D2")

    wb.save("chart_demo.xlsx")
    print("📊 Created chart demo: chart_demo.xlsx")

# Run advanced demo
# demo_advanced_excel()
```

---

## 🌐 **Web Scraping & API Automation**

### **Web Scraping with BeautifulSoup and Requests**

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Optional
import time
import json
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 1. Web Scraper Class
class WebScraper:
    """Professional web scraping class with error handling"""

    def __init__(self, delay: float = 1.0, timeout: int = 30):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.results = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Get and parse web page"""
        try:
            self.logger.info(f"Scraping: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            time.sleep(self.delay)  # Respectful scraping
            return soup

        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None

    def scrape_table_data(self, url: str, table_selector: str = "table") -> List[Dict]:
        """Scrape data from HTML tables"""
        soup = self.get_page(url)
        if not soup:
            return []

        tables = soup.select(table_selector)
        all_data = []

        for table in tables:
            headers = [th.get_text().strip() for th in table.select('th')]

            for row in table.select('tr')[1:]:  # Skip header row
                cells = row.select('td')
                if len(cells) == len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells):
                        row_data[headers[i]] = cell.get_text().strip()
                    all_data.append(row_data)

        return all_data

    def scrape_product_listings(self, url: str,
                              product_selector: str = ".product-item",
                              title_selector: str = ".title",
                              price_selector: str = ".price") -> List[Dict]:
        """Scrape product listings from e-commerce sites"""
        soup = self.get_page(url)
        if not soup:
            return []

        products = soup.select(product_selector)
        product_data = []

        for product in products:
            title_elem = product.select_one(title_selector)
            price_elem = product.select_one(price_selector)

            if title_elem and price_elem:
                product_data.append({
                    'title': title_elem.get_text().strip(),
                    'price': price_elem.get_text().strip(),
                    'url': product.find('a')['href'] if product.find('a') else None
                })

        return product_data

    def scrape_news_articles(self, url: str,
                           article_selector: str = "article",
                           title_selector: str = "h2, h3",
                           link_selector: str = "a",
                           summary_selector: str = "p") -> List[Dict]:
        """Scrape news articles"""
        soup = self.get_page(url)
        if not soup:
            return []

        articles = soup.select(article_selector)
        news_data = []

        for article in articles:
            title_elem = article.select_one(title_selector)
            link_elem = article.select_one(link_selector)
            summary_elem = article.select_one(summary_selector)

            if title_elem and link_elem:
                news_data.append({
                    'title': title_elem.get_text().strip(),
                    'url': urljoin(url, link_elem['href']),
                    'summary': summary_elem.get_text().strip() if summary_elem else None,
                    'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
                })

        return news_data

    def scrape_real_estate_listings(self, url: str) -> List[Dict]:
        """Scrape real estate listings"""
        soup = self.get_page(url)
        if not soup:
            return []

        listings = soup.select('[class*="property"], [class*="listing"]')
        property_data = []

        for listing in listings:
            # Extract common property fields
            title_elem = listing.select_one('h2, h3, .title, .address')
            price_elem = listing.select_one('.price, [class*="price"]')
            beds_elem = listing.select_one('.beds, [class*="bed"]')
            baths_elem = listing.select_one('.baths, [class*="bath"]')
            sqft_elem = listing.select_one('.sqft, [class*="sqft"]')
            link_elem = listing.select_one('a')

            property_data.append({
                'title': title_elem.get_text().strip() if title_elem else None,
                'price': price_elem.get_text().strip() if price_elem else None,
                'bedrooms': beds_elem.get_text().strip() if beds_elem else None,
                'bathrooms': baths_elem.get_text().strip() if baths_elem else None,
                'square_feet': sqft_elem.get_text().strip() if sqft_elem else None,
                'url': urljoin(url, link_elem['href']) if link_elem and 'href' in link_elem.attrs else None
            })

        return property_data

# 2. Concurrent Web Scraping
class ConcurrentWebScraper:
    """Concurrent web scraping for multiple URLs"""

    def __init__(self, max_workers: int = 5, delay: float = 0.5):
        self.max_workers = max_workers
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape single URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            time.sleep(self.delay)  # Be respectful

            return {
                'url': url,
                'status_code': response.status_code,
                'title': soup.title.string if soup.title else None,
                'headers_count': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                'links_count': len(soup.find_all('a')),
                'images_count': len(soup.find_all('img')),
                'paragraphs_count': len(soup.find_all('p')),
                'word_count': len(soup.get_text().split()),
                'success': True
            }

        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'success': False
            }

    def scrape_multiple_urls(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs concurrently"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.scrape_url, url): url for url in urls}

            for future in as_completed(future_to_url):
                result = future.result()
                results.append(result)

        return results

# 3. API Integration
class APIIntegrator:
    """Professional API integration class"""

    def __init__(self, base_url: str = "", headers: Dict = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(headers or {})

        # Setup rate limiting
        self.last_request_time = 0
        self.rate_limit = 1.0  # seconds between requests

    def rate_limited_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make rate-limited API request"""
        # Wait for rate limit
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()

            self.last_request_time = time.time()

            # Try to parse JSON, fallback to text
            try:
                return {
                    'status_code': response.status_code,
                    'data': response.json(),
                    'success': True
                }
            except json.JSONDecodeError:
                return {
                    'status_code': response.status_code,
                    'data': response.text,
                    'success': True
                }

        except requests.RequestException as e:
            return {
                'endpoint': endpoint,
                'error': str(e),
                'success': False
            }

    def get_weather_data(self, api_key: str, city: str) -> Dict[str, Any]:
        """Get weather data from weather API"""
        endpoint = f"/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        return self.rate_limited_request('GET', endpoint)

    def get_news_data(self, api_key: str, query: str, country: str = "us") -> Dict[str, Any]:
        """Get news data from news API"""
        endpoint = f"/v2/everything?q={query}&country={country}&apiKey={api_key}"
        return self.rate_limited_request('GET', endpoint)

    def post_form_data(self, endpoint: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Post form data to API"""
        return self.rate_limited_request('POST', endpoint, data=form_data)

# 4. Data Export and Processing
class DataExporter:
    """Export scraped data to various formats"""

    @staticmethod
    def export_to_csv(data: List[Dict], filename: str) -> None:
        """Export data to CSV"""
        if not data:
            print("⚠️ No data to export")
            return

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"📄 Exported {len(data)} records to {filename}")

    @staticmethod
    def export_to_json(data: List[Dict], filename: str) -> None:
        """Export data to JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"📄 Exported {len(data)} records to {filename}")

    @staticmethod
    def export_to_excel(data: List[Dict], filename: str, sheet_name: str = "Data") -> None:
        """Export data to Excel"""
        if not data:
            print("⚠️ No data to export")
            return

        df = pd.DataFrame(data)
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"📄 Exported {len(data)} records to {filename}")

    @staticmethod
    def clean_data(data: List[Dict]) -> List[Dict]:
        """Clean scraped data"""
        cleaned_data = []

        for item in data:
            cleaned_item = {}
            for key, value in item.items():
                if isinstance(value, str):
                    # Remove extra whitespace and newlines
                    cleaned_value = re.sub(r'\s+', ' ', value.strip())
                    cleaned_item[key] = cleaned_value
                else:
                    cleaned_item[key] = value
            cleaned_data.append(cleaned_item)

        return cleaned_data

# Demo web scraping and API integration
def demo_web_scraping():
    """Demonstrate web scraping capabilities"""
    print("🕷️ Web Scraping & API Integration Demo")
    print("=" * 50)

    # Note: These are example URLs - replace with real sites for actual scraping

    # Demo concurrent scraping
    print("🔄 Concurrent Web Scraping:")
    concurrent_scraper = ConcurrentWebScraper(max_workers=3)

    # Example URLs (these are placeholders)
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3"
    ]

    print("  Note: Using example URLs for demonstration")
    print("  In real usage, replace with actual websites")

    # Demo API integration
    print("\n🌐 API Integration:")
    api = APIIntegrator(base_url="https://api.openweathermap.org/data/2.5")

    print("  Weather API integration ready (requires API key)")
    print("  Supports: GET, POST requests with rate limiting")

    # Demo data export
    print("\n📊 Data Export:")
    sample_data = [
        {"name": "Product 1", "price": 29.99, "category": "Electronics"},
        {"name": "Product 2", "price": 15.50, "category": "Books"},
        {"name": "Product 3", "price": 89.99, "category": "Clothing"}
    ]

    # Export to different formats
    DataExporter.export_to_csv(sample_data, "products.csv")
    DataExporter.export_to_json(sample_data, "products.json")
    DataExporter.export_to_excel(sample_data, "products.xlsx")

    # Demo data cleaning
    dirty_data = [
        {"name": "  Product 1  ", "price": "29.99", "category": "Electronics\n"},
        {"name": "Product 2", "price": "15.50", "category": "  Books  "}
    ]

    clean_data = DataExporter.clean_data(dirty_data)
    print(f"  Cleaned {len(dirty_data)} records")

    print("✅ Web scraping demo completed!")

# Run the demo
# demo_web_scraping()
```

### **Advanced Web Automation with Selenium**

```python
# Note: Selenium requires additional installation: pip install selenium
# And requires webdriver (ChromeDriver, FirefoxDriver, etc.)

"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

class AdvancedWebAutomator:
    '''Advanced web automation with Selenium'''

    def __init__(self, headless: bool = True):
        self.driver = None
        self.headless = headless
        self.wait_time = 10

    def start_browser(self) -> None:
        '''Start browser with options'''
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.maximize_window()
        print("🌐 Browser started")

    def navigate_to(self, url: str) -> None:
        '''Navigate to URL'''
        if not self.driver:
            self.start_browser()

        self.driver.get(url)
        print(f"📍 Navigated to: {url}")

    def wait_for_element(self, by: By, value: str, timeout: int = None) -> bool:
        '''Wait for element to be present'''
        timeout = timeout or self.wait_time
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return True
        except TimeoutException:
            return False

    def click_element(self, by: By, value: str) -> bool:
        '''Click element'''
        try:
            element = self.driver.find_element(by, value)
            element.click()
            print(f"🖱️ Clicked: {value}")
            return True
        except NoSuchElementException:
            print(f"❌ Element not found: {value}")
            return False

    def fill_form(self, form_data: Dict[str, str]) -> bool:
        '''Fill form with data'''
        success = True
        for field_name, value in form_data.items():
            try:
                # Try different selectors
                selectors = [
                    f"input[name='{field_name}']",
                    f"input[id='{field_name}']",
                    f"#{field_name}",
                    f"input[placeholder*='{field_name}']"
                ]

                filled = False
                for selector in selectors:
                    try:
                        element = self.driver.find_element(By.CSS_SELECTOR, selector)
                        element.clear()
                        element.send_keys(value)
                        print(f"✍️ Filled {field_name}: {value}")
                        filled = True
                        break
                    except NoSuchElementException:
                        continue

                if not filled:
                    print(f"⚠️ Could not fill field: {field_name}")
                    success = False

            except Exception as e:
                print(f"❌ Error filling {field_name}: {e}")
                success = False

        return success

    def take_screenshot(self, filename: str = "screenshot.png") -> None:
        '''Take screenshot'''
        if self.driver:
            self.driver.save_screenshot(filename)
            print(f"📸 Screenshot saved: {filename}")

    def get_page_source(self) -> str:
        '''Get page source'''
        return self.driver.page_source if self.driver else ""

    def execute_script(self, script: str) -> Any:
        '''Execute JavaScript'''
        if self.driver:
            return self.driver.execute_script(script)
        return None

    def close_browser(self) -> None:
        '''Close browser'''
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("❌ Browser closed")

# Demo advanced web automation
def demo_selenium_automation():
    '''Demonstrate Selenium automation'''
    print("🤖 Selenium Web Automation Demo")
    print("=" * 40)

    # This would require actual websites to test
    print("🔧 Selenium automation ready")
    print("  Features:")
    print("  - Browser automation")
    print("  - Form filling")
    print("  - Element interaction")
    print("  - Screenshot capture")
    print("  - JavaScript execution")

    print("  Note: Requires webdriver installation")
    print("  pip install selenium")
    print("  # Download ChromeDriver or FirefoxDriver")
"""

print("📝 Selenium automation example (commented out - requires webdriver)")
print("💡 Install selenium and webdriver for full functionality")
```

---

## 🔄 **Office Workflow Automation**

### **File Management Automation**

```python
import os
import shutil
import glob
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import tarfile
from typing import List, Dict, Any, Optional
import fnmatch
import hashlib

# 1. File Organization System
class FileOrganizer:
    """Automated file organization and management"""

    def __init__(self, base_directory: str):
        self.base_dir = Path(base_directory)
        self.organization_rules = {}

        # Default file type categories
        self.file_categories = {
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
            'spreadsheets': ['.xls', '.xlsx', '.csv', '.ods'],
            'presentations': ['.ppt', '.pptx', '.odp'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'],
            'videos': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c'],
            'data': ['.json', '.xml', '.yaml', '.yml', '.ini', '.cfg']
        }

    def add_organization_rule(self, pattern: str, target_folder: str) -> None:
        '''Add custom organization rule'''
        self.organization_rules[pattern] = target_folder

    def organize_files(self, source_directory: str = None) -> Dict[str, int]:
        '''Organize files by type'''
        source_dir = Path(source_directory) if source_directory else self.base_dir
        organization_stats = {}

        # Process all files in directory
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                target_folder = self._determine_target_folder(file_path, file_ext)

                if target_folder:
                    target_path = source_dir / target_folder / file_path.name

                    # Handle duplicate names
                    counter = 1
                    original_name = target_path.stem
                    while target_path.exists():
                        new_name = f"{original_name}_{counter}{target_path.suffix}"
                        target_path = target_dir / target_folder / new_name
                        counter += 1

                    # Move file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(target_path))

                    # Update stats
                    folder = target_folder
                    organization_stats[folder] = organization_stats.get(folder, 0) + 1

        return organization_stats

    def _determine_target_folder(self, file_path: Path, file_ext: str) -> Optional[str]:
        '''Determine target folder for file'''
        # Check custom rules first
        for pattern, folder in self.organization_rules.items():
            if fnmatch.fnmatch(file_path.name.lower(), pattern.lower()):
                return folder

        # Check default categories
        for category, extensions in self.file_categories.items():
            if file_ext in extensions:
                return category

        return 'miscellaneous'

    def cleanup_duplicates(self, directory: str = None,
                         dry_run: bool = True) -> List[Dict[str, Any]]:
        '''Find and optionally remove duplicate files'''
        source_dir = Path(directory) if directory else self.base_dir
        duplicate_info = []
        file_hashes = {}

        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                # Calculate file hash
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in file_hashes:
                    duplicate_info.append({
                        'original': file_hashes[file_hash],
                        'duplicate': str(file_path),
                        'size': file_path.stat().st_size
                    })

                    if not dry_run:
                        file_path.unlink()
                        print(f"🗑️ Removed duplicate: {file_path}")
                else:
                    file_hashes[file_hash] = str(file_path)

        return duplicate_info

    def create_backup(self, source_directory: str,
                     backup_location: str = None,
                     compression: str = 'zip') -> str:
        '''Create compressed backup'''
        source_path = Path(source_directory)
        backup_path = Path(backup_location) if backup_location else Path.cwd()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if compression == 'zip':
            backup_file = backup_path / f"{source_path.name}_backup_{timestamp}.zip"
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_path)
                        zipf.write(file_path, arcname)

        elif compression == 'tar.gz':
            backup_file = backup_path / f"{source_path.name}_backup_{timestamp}.tar.gz"
            with tarfile.open(backup_file, 'w:gz') as tarf:
                tarf.add(source_path, arcname=source_path.name)

        print(f"💾 Backup created: {backup_file}")
        return str(backup_file)

    def file_search(self, pattern: str, directory: str = None,
                   case_sensitive: bool = False) -> List[Path]:
        '''Search for files by pattern'''
        source_dir = Path(directory) if directory else self.base_dir

        matches = []
        if case_sensitive:
            matches = list(source_dir.rglob(pattern))
        else:
            # Case-insensitive search
            for file_path in source_dir.rglob('*'):
                if file_path.is_file() and pattern.lower() in file_path.name.lower():
                    matches.append(file_path)

        return matches

# 2. Document Processing System
class DocumentProcessor:
    '''Process and manipulate documents'''

    def __init__(self):
        self.supported_formats = {
            'text': ['.txt', '.md'],
            'pdf': ['.pdf'],
            'office': ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        '''Extract text from PDF (requires PyPDF2 or pdfplumber)'''
        try:
            # This would require: pip install PyPDF2 or pdfplumber
            # For now, return placeholder
            return "PDF text extraction requires PyPDF2 or pdfplumber"
        except Exception as e:
            return f"Error extracting PDF text: {e}"

    def create_summary_report(self, documents: List[str],
                            output_file: str = "summary_report.txt") -> None:
        '''Create summary of multiple documents'''
        summaries = []

        for doc_path in documents:
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Simple summary: first 200 characters
                summary = content[:200] + "..." if len(content) > 200 else content
                summaries.append(f"File: {os.path.basename(doc_path)}\n")
                summaries.append(f"Size: {len(content)} characters\n")
                summaries.append(f"Summary: {summary}\n")
                summaries.append("-" * 50 + "\n")

            except Exception as e:
                summaries.append(f"Error processing {doc_path}: {e}\n")

        # Write summary
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(summaries)

        print(f"📄 Summary report created: {output_file}")

    def batch_rename(self, directory: str, name_pattern: str = "file_{:03d}{}") -> None:
        '''Batch rename files in directory'''
        directory_path = Path(directory)
        files = sorted([f for f in directory_path.iterdir() if f.is_file()])

        for index, file_path in enumerate(files, 1):
            new_name = name_pattern.format(index, file_path.suffix)
            new_path = directory_path / new_name
            file_path.rename(new_path)
            print(f"📝 Renamed: {file_path.name} → {new_name}")

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        '''Extract file metadata'''
        path = Path(file_path)
        stat = path.stat()

        metadata = {
            'filename': path.name,
            'extension': path.suffix,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'accessed': datetime.fromtimestamp(stat.st_atime)
        }

        return metadata

# 3. Automated Report Generator
class ReportGenerator:
    '''Generate automated reports'''

    def __init__(self):
        self.report_templates = {}

    def generate_file_analysis_report(self, directory: str,
                                    output_file: str = "file_analysis.txt") -> None:
        '''Generate comprehensive file analysis report'''
        directory_path = Path(directory)

        # Analyze files
        file_stats = {
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'largest_files': [],
            'oldest_files': []
        }

        all_files = []

        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                all_files.append(file_path)
                file_stats['total_files'] += 1
                file_stats['total_size'] += file_path.stat().st_size

                # Track file types
                ext = file_path.suffix.lower()
                file_stats['file_types'][ext] = file_stats['file_types'].get(ext, 0) + 1

        # Find largest files
        file_stats['largest_files'] = sorted(all_files,
                                           key=lambda x: x.stat().st_size,
                                           reverse=True)[:10]

        # Find oldest files
        file_stats['oldest_files'] = sorted(all_files,
                                          key=lambda x: x.stat().st_mtime)[:10]

        # Generate report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FILE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Directory: {directory}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("SUMMARY:\n")
            f.write(f"Total Files: {file_stats['total_files']:,}\n")
            f.write(f"Total Size: {file_stats['total_size'] / 1024 / 1024:.2f} MB\n\n")

            f.write("FILE TYPES:\n")
            for ext, count in sorted(file_stats['file_types'].items()):
                percentage = (count / file_stats['total_files']) * 100
                f.write(f"{ext or 'no extension'}: {count} files ({percentage:.1f}%)\n")

            f.write("\nLARGEST FILES:\n")
            for file_path in file_stats['largest_files']:
                size_mb = file_path.stat().st_size / 1024 / 1024
                f.write(f"{file_path.name}: {size_mb:.2f} MB\n")

            f.write("\nOLDEST FILES:\n")
            for file_path in file_stats['oldest_files']:
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                f.write(f"{file_path.name}: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"📊 File analysis report created: {output_file}")

# Demo office workflow automation
def demo_office_automation():
    '''Demonstrate office workflow automation'''
    print("🏢 Office Workflow Automation Demo")
    print("=" * 40)

    # File organization demo
    print("📁 File Organization:")
    # This would work with actual directories
    print("  - Organize files by type")
    print("  - Remove duplicates")
    print("  - Create backups")
    print("  - Search files")

    # Document processing demo
    print("\n📄 Document Processing:")
    print("  - Extract text from PDFs")
    print("  - Create summary reports")
    print("  - Batch rename files")
    print("  - Extract metadata")

    # Report generation demo
    print("\n📊 Report Generation:")
    print("  - File analysis reports")
    print("  - Custom report templates")
    print("  - Automated scheduling")

    # Create sample data for demo
    sample_files = [
        "document1.txt",
        "spreadsheet1.xlsx",
        "image1.jpg",
        "archive1.zip"
    ]

    # Simulate file organization
    print(f"\n🔧 Organizing {len(sample_files)} sample files:")
    for filename in sample_files:
        print(f"  Moving {filename} to appropriate folder")

    print("✅ Office automation demo completed!")

# Run the demo
# demo_office_automation()
```

---

## ⏰ **Scheduled Tasks & Automation**

### **Task Scheduling System**

```python
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Callable, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

# 1. Task Scheduler Class
class TaskScheduler:
    '''Professional task scheduling system'''

    def __init__(self):
        self.tasks = {}
        self.running = False
        self.scheduler_thread = None
        self.task_results = {}

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def add_task(self, task_id: str, func: Callable,
                schedule_info: str, **kwargs) -> None:
        '''Add task to scheduler'''
        if schedule_info.startswith('every'):
            # Parse "every X minutes/hours/days"
            if 'minute' in schedule_info:
                minutes = int(schedule_info.split()[1])
                schedule.every(minutes).minutes.do(self._execute_task, task_id, func, **kwargs)
            elif 'hour' in schedule_info:
                hours = int(schedule_info.split()[1])
                schedule.every(hours).hours.do(self._execute_task, task_id, func, **kwargs)
            elif 'day' in schedule_info:
                days = int(schedule_info.split()[1])
                schedule.every(days).days.do(self._execute_task, task_id, func, **kwargs)
        else:
            # Parse time like "14:30" or "2:30 PM"
            schedule.every().day.at(schedule_info).do(self._execute_task, task_id, func, **kwargs)

        self.tasks[task_id] = {
            'func': func,
            'schedule': schedule_info,
            'kwargs': kwargs,
            'last_run': None,
            'next_run': None,
            'status': 'scheduled'
        }

        self.logger.info(f"📅 Added task: {task_id} ({schedule_info})")

    def _execute_task(self, task_id: str, func: Callable, **kwargs) -> None:
        '''Execute task with error handling'''
        self.logger.info(f"🚀 Executing task: {task_id}")

        try:
            start_time = datetime.now()
            result = func(**kwargs)
            end_time = datetime.now()

            execution_time = (end_time - start_time).total_seconds()

            # Store result
            self.task_results[task_id] = {
                'result': result,
                'execution_time': execution_time,
                'start_time': start_time,
                'end_time': end_time,
                'status': 'success'
            }

            # Update task info
            self.tasks[task_id]['last_run'] = end_time
            self.tasks[task_id]['status'] = 'success'

            self.logger.info(f"✅ Task {task_id} completed in {execution_time:.2f}s")

        except Exception as e:
            # Store error
            self.task_results[task_id] = {
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'start_time': start_time,
                'end_time': datetime.now(),
                'status': 'error'
            }

            self.tasks[task_id]['status'] = 'error'

            self.logger.error(f"❌ Task {task_id} failed: {e}")

    def start_scheduler(self) -> None:
        '''Start the scheduler'''
        if self.running:
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("⏰ Task scheduler started")

    def stop_scheduler(self) -> None:
        '''Stop the scheduler'''
        self.running = False
        schedule.clear()

        if self.scheduler_thread:
            self.scheduler_thread.join()

        self.logger.info("⏹️ Task scheduler stopped")

    def _run_scheduler(self) -> None:
        '''Main scheduler loop'''
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def get_task_status(self) -> Dict[str, Any]:
        '''Get status of all tasks'''
        status = {}
        for task_id, task_info in self.tasks.items():
            status[task_id] = {
                'schedule': task_info['schedule'],
                'last_run': task_info['last_run'].isoformat() if task_info['last_run'] else None,
                'status': task_info['status'],
                'next_run': str(schedule.next_run()) if task_id in [t for t in schedule.jobs] else None
            }
        return status

    def get_task_results(self, task_id: str) -> Dict[str, Any]:
        '''Get results for specific task'''
        return self.task_results.get(task_id, {})

# 2. System Monitoring Tasks
class SystemMonitor:
    '''Automated system monitoring tasks'''

    def __init__(self):
        self.monitoring_data = []

    def check_disk_space(self, threshold: int = 80) -> Dict[str, Any]:
        '''Check disk space usage'''
        import shutil
        total, used, free = shutil.disk_usage("/")

        usage_percent = (used / total) * 100

        result = {
            'timestamp': datetime.now().isoformat(),
            'total_gb': total // (1024**3),
            'used_gb': used // (1024**3),
            'free_gb': free // (1024**3),
            'usage_percent': usage_percent,
            'status': 'warning' if usage_percent > threshold else 'normal'
        }

        self.monitoring_data.append(result)

        if usage_percent > threshold:
            print(f"⚠️ Disk space warning: {usage_percent:.1f}% used")

        return result

    def check_memory_usage(self) -> Dict[str, Any]:
        '''Check memory usage'''
        import psutil

        memory = psutil.virtual_memory()

        result = {
            'timestamp': datetime.now().isoformat(),
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'usage_percent': memory.percent,
            'status': 'warning' if memory.percent > 80 else 'normal'
        }

        self.monitoring_data.append(result)

        if memory.percent > 80:
            print(f"⚠️ Memory usage warning: {memory.percent:.1f}% used")

        return result

    def generate_monitoring_report(self) -> str:
        '''Generate monitoring report'''
        if not self.monitoring_data:
            return "No monitoring data available"

        # Analyze data
        latest = self.monitoring_data[-1]
        disk_warnings = len([d for d in self.monitoring_data if d.get('status') == 'warning'])

        report = f"""
SYSTEM MONITORING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

LATEST STATUS:
- Disk Usage: {latest.get('usage_percent', 0):.1f}%
- Memory Usage: {latest.get('memory_percent', 0):.1f}%

SUMMARY:
- Total monitoring records: {len(self.monitoring_data)}
- Warning conditions: {disk_warnings}
"""

        return report

# 3. Automated Backup System
class BackupScheduler:
    '''Automated backup scheduling'''

    def __init__(self, backup_directory: str = "./backups"):
        self.backup_dir = Path(backup_directory)
        self.backup_dir.mkdir(exist_ok=True)

    def backup_directory(self, source_directory: str,
                        backup_name: str = None) -> str:
        '''Backup a directory'''
        source_path = Path(source_directory)

        if not backup_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{source_path.name}_backup_{timestamp}"

        backup_path = self.backup_dir / f"{backup_name}.zip"

        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)

        print(f"💾 Backup created: {backup_path}")
        return str(backup_path)

    def cleanup_old_backups(self, days_to_keep: int = 30) -> None:
        '''Clean up old backup files'''
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        removed_count = 0
        for backup_file in self.backup_dir.glob("*.zip"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                removed_count += 1
                print(f"🗑️ Removed old backup: {backup_file.name}")

        print(f"🧹 Cleaned up {removed_count} old backups")

# 4. Email Automation (Requires email configuration)
class EmailAutomator:
    '''Automated email sending'''

    def __init__(self, smtp_server: str = "smtp.gmail.com", port: int = 587):
        self.smtp_server = smtp_server
        self.port = port
        self.smtp_user = None
        self.smtp_password = None

    def configure_smtp(self, username: str, password: str) -> None:
        '''Configure SMTP credentials'''
        self.smtp_user = username
        self.smtp_password = password

    def send_report_email(self, to_email: str, subject: str,
                         body: str, attachments: List[str] = None) -> bool:
        '''Send email report'''
        if not self.smtp_user or not self.smtp_password:
            print("❌ SMTP not configured")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.mime.base import MIMEBase
            from email import encoders

            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            # Add attachments
            if attachments:
                for file_path in attachments:
                    with open(file_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())

                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f"attachment; filename= {os.path.basename(file_path)}",
                    )
                    msg.attach(part)

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.smtp_user, to_email, text)
            server.quit()

            print(f"📧 Email sent to {to_email}")
            return True

        except Exception as e:
            print(f"❌ Email sending failed: {e}")
            return False

# Demo scheduled automation
def demo_scheduled_automation():
    '''Demonstrate scheduled automation'''
    print("⏰ Scheduled Automation Demo")
    print("=" * 40)

    # Create scheduler
    scheduler = TaskScheduler()

    # Add sample tasks
    def daily_report():
        return f"Daily report generated at {datetime.now()}"

    def system_check():
        return "System check completed"

    def backup_files():
        return "Backup completed"

    scheduler.add_task("daily_report", daily_report, "09:00")
    scheduler.add_task("system_check", system_check, "every 30 minutes")
    scheduler.add_task("backup_files", backup_files, "every 2 hours")

    print("📅 Scheduled tasks:")
    for task_id, task_info in scheduler.tasks.items():
        print(f"  {task_id}: {task_info['schedule']}")

    # System monitoring demo
    print("\n🔍 System Monitoring:")
    monitor = SystemMonitor()

    disk_result = monitor.check_disk_space()
    memory_result = monitor.check_memory_usage()

    print(f"  Disk: {disk_result['usage_percent']:.1f}% ({disk_result['status']})")
    print(f"  Memory: {memory_result['usage_percent']:.1f}% ({memory_result['status']})")

    # Backup system demo
    print("\n💾 Backup System:")
    backup = BackupScheduler()
    print("  Backup system initialized")
    print("  Automatic cleanup of old backups")

    # Email automation demo
    print("\n📧 Email Automation:")
    email = EmailAutomator()
    print("  Email automation ready")
    print("  Configure SMTP for full functionality")

    print("✅ Scheduled automation demo completed!")
    print("💡 Start scheduler with: scheduler.start_scheduler()")

# Run the demo
# demo_scheduled_automation()
```

---

## 🎉 **Congratulations!**

You've mastered **Python Automation Projects** for real-world productivity!

### **What You've Accomplished:**

✅ **Excel Automation** - File operations, data processing, report generation, formulas  
✅ **Web Scraping** - BeautifulSoup, concurrent scraping, API integration, data export  
✅ **Office Workflow** - File organization, document processing, system monitoring  
✅ **Scheduled Tasks** - Task scheduling, system monitoring, backup automation, email

### **Your Automation Expertise:**

🎯 **Data Processing** - Automate spreadsheet and document workflows  
🎯 **Web Automation** - Scrape websites and integrate APIs efficiently  
🎯 **System Administration** - Organize files, monitor systems, create backups  
🎯 **Business Intelligence** - Generate reports, schedule tasks, automate decisions  
🎯 **Productivity Enhancement** - Build tools to save time and reduce manual work

### **Next Steps:**

🚀 **Build Custom Solutions** - Create automation for your specific needs  
🚀 **Scale Automation** - Use advanced techniques for large-scale operations  
🚀 **Integrate Systems** - Connect different automation tools for complex workflows  
🚀 **Monitor & Optimize** - Track automation performance and improve efficiency

**🔗 Continue Your Journey:** Move to `python_ai_tools_integration_complete_guide.md` for AI-powered automation!

---

## _Automation isn't about replacing humans—it's about empowering people to focus on what matters most!_ 🤖⚡✨

## 🔍 COMMON CONFUSIONS & MISTAKES

### 1. Web Scraping Ethics and Legal Issues

**❌ Mistake:** Scraping websites without checking robots.txt or terms of service
**✅ Solution:** Always respect website policies and implement respectful scraping

```python
import requests
from urllib.robotparser import RobotFileParser

def check_scraping_allowed(url):
    """Check if scraping is allowed for a given URL"""
    rp = RobotFileParser()
    rp.set_url(f"{url}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return True  # Default to allowed if robots.txt not accessible
```

### 2. Excel File Corruption Prevention

**❌ Mistake:** Not handling file locks or concurrent access to Excel files
**✅ Solution:** Implement proper file handling and access control

```python
import openpyxl
from contextlib import contextmanager
import time

@contextmanager
def safe_excel_access(filepath, mode='r'):
    """Safely access Excel file with proper error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if mode == 'r':
                wb = openpyxl.load_workbook(filepath, read_only=True)
            else:
                wb = openpyxl.load_workbook(filepath, keep_vba=True)
            yield wb
            break
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait and retry
                continue
            raise Exception("File is locked or in use by another process")
```

### 3. API Rate Limiting Overlook

**❌ Mistake:** Making unlimited API calls without rate limiting
**✅ Solution:** Implement rate limiting and exponential backoff

```python
import time
import functools

def rate_limiter(max_calls=60, time_window=60):
    """Decorator to implement rate limiting for API calls"""
    def decorator(func):
        calls = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls outside the time window
            calls[:] = [call_time for call_time in calls if now - call_time < time_window]

            if len(calls) >= max_calls:
                sleep_time = time_window - (now - calls[0])
                time.sleep(sleep_time)

            result = func(*args, **kwargs)
            calls.append(now)
            return result

        return wrapper
    return decorator
```

### 4. Error Handling in Automation Scripts

**❌ Mistake:** Not handling errors gracefully in long-running automation
**✅ Solution:** Implement comprehensive error handling and logging

```python
import logging
from functools import wraps

def setup_automation_logging():
    """Setup logging for automation scripts"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('automation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def safe_automation_task(func):
    """Decorator to safely execute automation tasks with error handling"""
    logger = setup_automation_logging()

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Starting task: {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Task completed successfully: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Task failed: {func.__name__} - {str(e)}")
            raise

    return wrapper
```

### 5. Data Privacy in Automation

**❌ Mistake:** Not handling sensitive data properly in automation scripts
**✅ Solution:** Implement data sanitization and secure storage

```python
import hashlib
import re
from typing import Any

class DataSanitizer:
    """Handle sensitive data in automation scripts"""

    @staticmethod
    def sanitize_email(email):
        """Remove or mask sensitive email information"""
        if not email or '@' not in email:
            return "masked@example.com"

        username, domain = email.split('@', 1)
        # Keep first character and mask rest
        masked_username = username[0] + '*' * (len(username) - 1)
        return f"{masked_username}@{domain}"

    @staticmethod
    def hash_sensitive_data(data: str, salt: str = "automation_salt") -> str:
        """Create hash of sensitive data for logging without exposing actual values"""
        return hashlib.sha256((data + salt).encode()).hexdigest()[:16]
```

### 6. File Path Handling Issues

**❌ Mistake:** Using hardcoded file paths that don't work across different systems
**✅ Solution:** Use pathlib and proper path handling

```python
from pathlib import Path
import os

def get_safe_project_path(*path_parts):
    """Get safe project path that works across different systems"""
    current_dir = Path(__file__).parent
    return current_dir / Path(*path_parts)

# Use relative paths instead of absolute
DATA_PATH = get_safe_project_path("data")
OUTPUT_PATH = get_safe_project_path("output")
LOG_PATH = get_safe_project_path("logs")
```

### 7. Memory Management in Large Data Processing

**❌ Mistake:** Loading entire large datasets into memory at once
**✅ Solution:** Use chunked processing and generators

```python
import pandas as pd
from typing import Iterator

def process_large_excel_file(filepath: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
    """Process large Excel files in chunks to manage memory"""
    try:
        # Try to read Excel in chunks
        xl_file = pd.ExcelFile(filepath)
        for sheet_name in xl_file.sheet_names:
            for chunk in pd.read_excel(xl_file, sheet_name=sheet_name, chunksize=chunk_size):
                yield chunk
    except Exception as e:
        print(f"Error processing file: {e}")
```

### 8. Selenium WebDriver Best Practices

**❌ Mistake:** Not properly managing WebDriver instances and browser cleanup
**✅ Solution:** Use proper browser management and cleanup

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from contextlib import contextmanager

@contextmanager
def managed_browser(headless=True, timeout=30):
    """Context manager for proper browser lifecycle management"""
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(timeout)

    try:
        yield driver
    finally:
        driver.quit()
```

---

## 📝 MICRO-QUIZ (80% MASTERY REQUIRED)

**Instructions:** Answer all questions. You need 5/6 correct (80%) to pass.

### Question 1: Web Scraping Ethics

What is the most important consideration when web scraping?
a) Scrape as much data as possible quickly
b) Respect website terms of service and robots.txt
c) Use the fastest scraping method available
d) Scrape only during business hours

**Correct Answer:** b) Respect website terms of service and robots.txt

### Question 2: Excel File Handling

What is the best practice for handling Excel files in automation?
a) Always use the latest version of openpyxl
b) Implement proper file locking and error handling
c) Load all data into memory at once for speed
d) Use absolute file paths for consistency

**Correct Answer:** b) Implement proper file locking and error handling

### Question 3: API Rate Limiting

Why is rate limiting important in automation scripts?
a) It's not important for API calls
b) To prevent overwhelming external services and avoid being blocked
c) To make automation scripts run faster
d) To reduce server costs only

**Correct Answer:** b) To prevent overwhelming external services and avoid being blocked

### Question 4: Data Privacy

When processing sensitive data in automation scripts, what should you always do?
a) Log all data for debugging purposes
b) Sanitize or mask sensitive information
c) Store passwords in plain text for easy access
d) Share data with all team members for transparency

**Correct Answer:** b) Sanitize or mask sensitive information

### Question 5: Error Handling

What is the best approach for error handling in long-running automation scripts?
a) Let errors crash the script and fix manually
b) Implement comprehensive error handling and logging
c) Ignore errors to keep the script running
d) Only handle critical errors, ignore warnings

**Correct Answer:** b) Implement comprehensive error handling and logging

### Question 6: File Path Management

What is the most reliable way to handle file paths in cross-platform automation?
a) Use hardcoded absolute paths
b) Use relative paths with pathlib
c) Only use Windows-style paths
d) Use environment variables for all paths

**Correct Answer:** b) Use relative paths with pathlib

---

## 🤔 REFLECTION PROMPTS

### 1. Concept Understanding

How would you explain the balance between automation efficiency and system resource usage to a non-technical manager? What examples would illustrate this trade-off?

**Reflection Focus:** Consider both productivity gains and resource costs. Think about scalable automation solutions that don't overburden systems.

### 2. Real-World Application

Think about a repetitive task you do regularly. How could Python automation improve this process? What challenges might you face in implementing the automation?

**Reflection Focus:** Apply automation concepts to personal or professional workflows. Consider both technical feasibility and practical implementation.

### 3. Future Evolution

How do you think automation will change in the next 5 years? What new opportunities and challenges might arise with AI integration and cloud services?

**Reflection Focus:** Consider emerging technologies, ethical implications, and the evolution of human-automation collaboration. Think about both positive and negative impacts.

---

## ⚡ MINI SPRINT PROJECT (20-30 minutes)

### Project: Automated Report Generator

Build a simple automation script that processes data and generates formatted reports.

**Objective:** Create a functional automation solution for common reporting tasks.

**Time Investment:** 20-30 minutes
**Difficulty Level:** Beginner to Intermediate
**Skills Practiced:** File processing, data manipulation, report formatting, error handling

### Step-by-Step Implementation

**Step 1: Data Processing Module (8 minutes)**

```python
# report_generator.py
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

class DataProcessor:
    def __init__(self):
        self.data = None
        self.processed_data = None

    def load_sample_data(self):
        """Load sample data for demonstration"""
        self.data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', '2024-01-15', freq='D'),
            'sales': [100 + i*10 + (i%3)*20 for i in range(15)],
            'expenses': [50 + i*5 + (i%4)*15 for i in range(15)],
            'region': ['North', 'South', 'East', 'West'] * 4 + ['North']
        })
        return self.data

    def process_data(self):
        """Process and analyze the data"""
        if self.data is None:
            raise ValueError("No data loaded")

        # Calculate profit
        self.data['profit'] = self.data['sales'] - self.data['expenses']

        # Calculate daily metrics
        daily_summary = self.data.groupby('date').agg({
            'sales': 'sum',
            'expenses': 'sum',
            'profit': 'sum'
        }).reset_index()

        # Calculate moving averages
        daily_summary['profit_ma3'] = daily_summary['profit'].rolling(3).mean()

        # Regional analysis
        regional_summary = self.data.groupby('region').agg({
            'sales': 'sum',
            'expenses': 'sum',
            'profit': 'sum'
        }).reset_index()

        self.processed_data = {
            'daily': daily_summary,
            'regional': regional_summary,
            'summary_stats': {
                'total_sales': self.data['sales'].sum(),
                'total_expenses': self.data['expenses'].sum(),
                'total_profit': self.data['profit'].sum(),
                'avg_daily_profit': self.data['profit'].mean(),
                'profit_margin': (self.data['profit'].sum() / self.data['sales'].sum()) * 100
            }
        }

        return self.processed_data
```

**Step 2: Report Generation (10 minutes)**

```python
class ReportGenerator:
    def __init__(self, processor: DataProcessor):
        self.processor = processor

    def generate_text_report(self, output_path: str = "report.txt"):
        """Generate a formatted text report"""
        data = self.processor.processed_data

        if not data:
            raise ValueError("No processed data available")

        report_lines = []
        report_lines.append("AUTOMATED BUSINESS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary section
        stats = data['summary_stats']
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Sales: ${stats['total_sales']:,.2f}")
        report_lines.append(f"Total Expenses: ${stats['total_expenses']:,.2f}")
        report_lines.append(f"Total Profit: ${stats['total_profit']:,.2f}")
        report_lines.append(f"Average Daily Profit: ${stats['avg_daily_profit']:,.2f}")
        report_lines.append(f"Profit Margin: {stats['profit_margin']:.1f}%")
        report_lines.append("")

        # Daily trends
        report_lines.append("DAILY PERFORMANCE TRENDS")
        report_lines.append("-" * 30)
        daily_data = data['daily']

        for _, row in daily_data.tail(5).iterrows():
            report_lines.append(
                f"{row['date'].strftime('%Y-%m-%d')}: "
                f"Profit ${row['profit']:,.2f} "
                f"(3-day avg: ${row['profit_ma3']:,.2f})"
            )
        report_lines.append("")

        # Regional analysis
        report_lines.append("REGIONAL PERFORMANCE")
        report_lines.append("-" * 25)
        regional_data = data['regional']

        for _, row in regional_data.iterrows():
            margin = (row['profit'] / row['sales']) * 100
            report_lines.append(
                f"{row['region']}: "
                f"${row['profit']:,.2f} profit "
                f"({margin:.1f}% margin)"
            )

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        return output_path

    def generate_json_export(self, output_path: str = "data_export.json"):
        """Export processed data as JSON for further analysis"""
        data = self.processor.processed_data

        # Convert to JSON-serializable format
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'summary_stats': data['summary_stats'],
            'daily_data': data['daily'].to_dict('records'),
            'regional_data': data['regional'].to_dict('records')
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        return output_path
```

**Step 3: Main Automation Script (7 minutes)**

```python
# main.py
import sys
import time
from pathlib import Path
from report_generator import DataProcessor, ReportGenerator

def main():
    print("🤖 Starting Automated Report Generation")
    print("=" * 40)

    try:
        # Initialize processor
        print("📊 Loading and processing data...")
        processor = DataProcessor()
        processor.load_sample_data()
        processor.process_data()

        # Initialize report generator
        generator = ReportGenerator(processor)

        # Generate reports
        print("📄 Generating text report...")
        text_report_path = generator.generate_text_report("automated_report.txt")
        print(f"✅ Text report saved: {text_report_path}")

        print("📋 Generating JSON export...")
        json_export_path = generator.generate_json_export("data_export.json")
        print(f"✅ JSON export saved: {json_export_path}")

        # Display summary
        stats = processor.processed_data['summary_stats']
        print("\n📈 REPORT SUMMARY")
        print("-" * 20)
        print(f"Total Sales: ${stats['total_sales']:,.2f}")
        print(f"Total Profit: ${stats['total_profit']:,.2f}")
        print(f"Profit Margin: {stats['profit_margin']:.1f}%")

        print("\n✅ Automation completed successfully!")

    except Exception as e:
        print(f"❌ Error in automation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n⏱️  Total execution time: {end_time - start_time:.2f} seconds")
```

**Step 4: Test and Validation (5 minutes)**

```python
# test_automation.py
import unittest
import os
from report_generator import DataProcessor, ReportGenerator

class TestAutomation(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.processor.load_sample_data()
        self.processor.process_data()
        self.generator = ReportGenerator(self.processor)

    def test_data_processing(self):
        """Test that data processing works correctly"""
        self.assertIsNotNone(self.processor.processed_data)
        self.assertIn('summary_stats', self.processor.processed_data)
        self.assertIn('daily', self.processor.processed_data)

    def test_report_generation(self):
        """Test that reports generate without errors"""
        text_path = self.generator.generate_text_report("test_report.txt")
        json_path = self.generator.generate_json_export("test_export.json")

        # Check files were created
        self.assertTrue(os.path.exists(text_path))
        self.assertTrue(os.path.exists(json_path))

        # Clean up
        os.remove(text_path)
        os.remove(json_path)

    def test_profit_calculation(self):
        """Test that profit calculations are correct"""
        stats = self.processor.processed_data['summary_stats']
        self.assertGreater(stats['total_profit'], 0)
        self.assertGreater(stats['profit_margin'], 0)

if __name__ == "__main__":
    # Run tests
    print("🧪 Running automation tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
```

### Success Criteria

- [ ] Successfully processes sample data and calculates metrics
- [ ] Generates formatted text reports with business insights
- [ ] Exports data in JSON format for further analysis
- [ ] Handles errors gracefully with proper logging
- [ ] Demonstrates automation efficiency compared to manual processes
- [ ] Includes basic testing to ensure reliability

### Test Your Implementation

1. Run the main script: `python main.py`
2. Check the generated files: `automated_report.txt` and `data_export.json`
3. Review the report content for accuracy
4. Run the test suite: `python test_automation.py`
5. Try modifying the data processing logic

### Quick Extensions (if time permits)

- Add Excel report generation with formatting
- Include data visualization (charts and graphs)
- Add email automation to send reports
- Implement scheduling for regular report generation
- Add data validation and error checking
- Create a simple web interface for report viewing

---

## 🏗️ FULL PROJECT EXTENSION (4-8 hours)

### Project: Comprehensive Business Automation Suite

Build a complete automation system for business processes including data collection, processing, analysis, and reporting.

**Objective:** Create a production-ready automation platform that handles multiple business workflows with monitoring and scheduling capabilities.

**Time Investment:** 4-8 hours
**Difficulty Level:** Advanced
**Skills Practiced:** System integration, data pipeline design, automation orchestration, monitoring

### Phase 1: Multi-Source Data Collection (1-2 hours)

**Features to Implement:**

- Web scraping from multiple sources
- API integration for external data
- File monitoring and processing
- Data validation and cleaning

### Phase 2: Automated Data Processing Pipeline (1-2 hours)

**Features to Implement:**

- ETL (Extract, Transform, Load) processes
- Data quality checks and reporting
- Business rule validation
- Error handling and recovery

### Phase 3: Report Generation and Distribution (1-2 hours)

**Features to Implement:**

- Multiple output formats (PDF, Excel, HTML)
- Email distribution with attachments
- Dashboard creation with metrics
- Scheduled report generation

### Phase 4: Monitoring and Alerting System (1-2 hours)

**Features to Implement:**

- System health monitoring
- Error alerting and notification
- Performance metrics tracking
- Automated backup and recovery

### Success Criteria

- [ ] Complete data collection from multiple sources
- [ ] Robust data processing with validation
- [ ] Professional report generation in multiple formats
- [ ] Reliable scheduling and monitoring system
- [ ] Error handling and recovery mechanisms
- [ ] User-friendly configuration and management interface

### Advanced Extensions

- **Machine Learning Integration:** Add predictive analytics to reports
- **Cloud Deployment:** Deploy on cloud platforms for scalability
- **API Gateway:** Create APIs for external system integration
- **Mobile Dashboard:** Build mobile-friendly monitoring interface
- **Advanced Analytics:** Add trend analysis and forecasting capabilities

## This project serves as a comprehensive demonstration of automation engineering skills, suitable for careers in business process automation, data engineering, or operations management.

## 🤝 Common Confusions & Misconceptions

### 1. Automation vs. Programming Confusion

**Misconception:** "Automation is just writing scripts to do repetitive tasks without real programming skill."
**Reality:** Automation requires the same programming skills as other development, plus additional considerations for reliability, scheduling, and error handling.
**Solution:** Approach automation as serious software development with the same quality standards and systematic approaches.

### 2. Small Task Automation Assumption

**Misconception:** "Automation is only useful for small, simple tasks and doesn't scale to complex business processes."
**Reality:** Well-designed automation systems can handle complex workflows and scale to handle enterprise-level business processes.
**Solution:** Learn to design automation systems with modular architecture that can handle complexity and scale appropriately.

### 3. One-Time Solution Thinking

**Misconception:** "Once I automate a process, I never need to think about it again."
**Reality:** Automated systems require maintenance, monitoring, updates, and adaptation as business needs change.
**Solution:** Design automation systems with maintainability, monitoring, and adaptability in mind from the beginning.

### 4. Error Handling Neglect

**Misconception:** "If my automation works most of the time, I don't need comprehensive error handling."
**Reality:** Automated systems must handle errors gracefully since they often run unattended and failure can have significant business impact.
**Solution:** Implement comprehensive error handling, monitoring, and recovery mechanisms for all automation systems.

### 5. Security Consideration Absence

**Misconception:** "Since automation runs automatically, I don't need to worry about security or access control."
**Reality:** Automated systems often have elevated privileges and access to sensitive data, making security even more critical.
**Solution:** Implement proper security measures, access controls, and audit trails for all automation systems.

### 6. User Interface Neglect

**Misconception:** "Automation doesn't need user interfaces since it runs automatically."
**Reality:** Automation systems need configuration interfaces, monitoring dashboards, and user interaction capabilities for management.
**Solution:** Design appropriate user interfaces for configuration, monitoring, and management of automation systems.

### 7. Performance Assumption

**Misconception:** "Automation systems don't need performance optimization since they run in the background."
**Reality:** Poor performance in automation can cause delays, resource issues, and business process disruption.
**Solution:** Design automation systems with performance considerations including efficient algorithms, resource management, and scalability.

### 8. Testing Oversight

**Misconception:** "Since automation does simple tasks, I don't need extensive testing."
**Reality:** Automation systems require comprehensive testing to ensure reliability, handle edge cases, and prevent business disruption.
**Solution:** Implement thorough testing including unit tests, integration tests, and automated testing for all automation components.

---

## 🧠 Micro-Quiz: Test Your Automation Mastery

### Question 1: Automation Design Priority

**What's the most important consideration when designing an automation system?**
A) Making the code as short as possible
B) Ensuring reliability, error handling, and proper monitoring
C) Using the most advanced Python features
D) Minimizing the number of automation steps

**Correct Answer:** B - Reliability, error handling, and monitoring are crucial for automation systems that often run unattended.

### Question 2: Error Handling Strategy

**Your automation script failed partway through execution. What's the best response?**
A) Assume it will work next time
B) Implement proper error handling, logging, and recovery mechanisms
C) Add more print statements for debugging
D) Run the automation manually

**Correct Answer:** B - Professional automation requires comprehensive error handling, logging, and recovery mechanisms.

### Question 3: Security Implementation

**Your automation system accesses sensitive business data. What's most important?**
A) Make the script run faster
B) Implement proper access controls, encryption, and audit trails
C) Use the newest Python features
D) Minimize the number of files created

**Correct Answer:** B - Security considerations are crucial for automation systems that access sensitive data or have elevated privileges.

### Question 4: Monitoring and Maintenance

**How should you monitor automated systems?**
A) Only check them when problems occur
B) Implement comprehensive logging, alerting, and monitoring systems
C) Assume they always work correctly
D) Monitor only during business hours

**Correct Answer:** B - Automated systems require comprehensive monitoring, logging, and alerting to ensure reliable operation.

### Question 5: Scalability Planning

**Your automation script works for small datasets but needs to handle large-scale data. What's the best approach?**
A) Assume it will scale automatically
B) Redesign with scalability considerations including efficient algorithms and resource management
C) Run multiple copies of the same script
D) Only process smaller datasets

**Correct Answer:** B - Scalable automation requires redesign with performance and resource considerations in mind.

### Question 6: User Interface Design

**Who should be able to configure and manage your automation system?**
A) Only the original developer
B) Any user who needs to use the automation
C) Appropriate users with proper training and access controls
D) No one should be able to configure it

**Correct Answer:** C - Automation systems need appropriate interfaces for authorized users with proper access controls and training.

---

## 💭 Reflection Prompts

### 1. Automation and Human Productivity

"Reflect on how automation changes the relationship between humans and repetitive work. How does this shift free up human creativity and problem-solving for more valuable activities? Consider how this pattern applies to other areas where technology augments human capabilities."

### 2. System Reliability and Trust

"Consider how automation requires building systems that people can trust to operate reliably without constant supervision. How does this influence your approach to designing, testing, and maintaining automated systems? What does this reveal about the responsibility that comes with building automation?"

### 3. Business Process Integration

"Think about how automation must integrate with existing business processes and workflows. How does this requirement for integration influence system design and development approaches? What does this teach about designing technology solutions that work within existing organizational contexts?"

---

## 🚀 Mini Sprint Project (1-3 hours)

### Comprehensive Workflow Automation System

**Objective:** Create a complete workflow automation system that demonstrates mastery of automation principles, error handling, and practical business application.

**Task Breakdown:**

1. **Workflow Analysis and Design (30 minutes):** Identify a repetitive business workflow and design automation system with proper error handling and monitoring
2. **Core Automation Implementation (75 minutes):** Build the automation system with proper modular design, error handling, and logging
3. **Monitoring and Management Interface (30 minutes):** Create interface for configuring, monitoring, and managing the automation system
4. **Testing and Validation (30 minutes):** Test the automation thoroughly including error scenarios and edge cases
5. **Documentation and Deployment (15 minutes):** Create documentation and deployment instructions for the automation system

**Success Criteria:**

- Complete automation system that handles realistic business workflow
- Demonstrates proper error handling, logging, and monitoring capabilities
- Includes user interface for configuration and management
- Shows practical application of automation principles in real-world scenario
- Provides foundation for understanding how automation scales to larger business processes

---

## 🏗️ Full Project Extension (10-25 hours)

### Enterprise Automation Platform and Management System

**Objective:** Build a comprehensive automation platform that demonstrates mastery of enterprise-level automation, workflow management, and business process integration through advanced system development.

**Extended Scope:**

#### Phase 1: Enterprise Automation Architecture (2-3 hours)

- **Comprehensive Workflow Analysis:** Define advanced automation capabilities including multi-step workflows, conditional logic, and business rule integration
- **Enterprise Automation Framework:** Design scalable framework for creating, managing, and deploying automation systems across business processes
- **Security and Compliance Planning:** Plan enterprise-grade security, access controls, audit trails, and regulatory compliance for automation systems
- **Performance and Scalability Design:** Design systems for handling enterprise-scale automation with optimal performance and resource management

#### Phase 2: Core Automation Engine Development (3-4 hours)

- **Workflow Engine Implementation:** Build sophisticated workflow engine with conditional logic, branching, and error recovery capabilities
- **Data Integration and Processing:** Implement comprehensive data integration with databases, APIs, file systems, and external services
- **Scheduling and Triggering System:** Create advanced scheduling, triggering, and event-driven automation with enterprise-grade reliability
- **Error Handling and Recovery:** Implement comprehensive error handling, logging, monitoring, and automated recovery mechanisms

#### Phase 3: Advanced Automation Features (3-4 hours)

- **Business Process Integration:** Build systems for integrating automation with existing business processes, workflows, and enterprise systems
- **Real-time Monitoring and Analytics:** Create comprehensive monitoring dashboards, analytics, and alerting for automation system health and performance
- **Configuration and Management Interface:** Build professional web interface for configuring, managing, and monitoring automation systems
- **API and Integration Capabilities:** Implement APIs for external system integration, webhook support, and third-party automation connectivity

#### Phase 4: Enterprise Quality and Operations (2-3 hours)

- **Comprehensive Testing Framework:** Build extensive testing including unit tests, integration tests, and automated workflow testing
- **Performance Optimization and Scaling:** Implement performance monitoring, optimization, and enterprise scaling capabilities
- **Security and Compliance Implementation:** Build enterprise security features including encryption, access controls, audit trails, and compliance reporting
- **Documentation and Training Systems:** Create comprehensive documentation, training materials, and operational procedures for enterprise deployment

#### Phase 5: Professional Deployment and Management (2-3 hours)

- **Containerized and Cloud Deployment:** Create enterprise deployment with containerization, cloud platforms, and enterprise infrastructure integration
- **High Availability and Disaster Recovery:** Implement enterprise-grade high availability, backup, and disaster recovery capabilities
- **Professional Operations Tools:** Build tools for system administration, maintenance, troubleshooting, and performance optimization
- **Compliance and Audit Capabilities:** Implement comprehensive compliance reporting, audit trails, and regulatory requirement satisfaction

#### Phase 6: Community and Professional Advancement (1-2 hours)

- **Open Source Automation Resources:** Plan contributions to open source automation tools and frameworks
- **Professional Automation Services:** Design professional consulting and implementation services for enterprise automation
- **Educational and Training Programs:** Create educational resources, training programs, and professional development in automation
- **Long-term Platform Evolution:** Plan for ongoing platform evolution, technology updates, and industry advancement

**Extended Deliverables:**

- Complete enterprise automation platform demonstrating mastery of business process automation and workflow management
- Professional-grade system with comprehensive workflow engine, monitoring, and enterprise deployment capabilities
- Advanced automation framework with modular design, security, and enterprise integration features
- Comprehensive testing, monitoring, and quality assurance systems for production automation deployment
- Professional documentation, training materials, and operational procedures for enterprise automation
- Professional consulting package and community contribution plan for ongoing automation advancement

**Impact Goals:**

- Demonstrate mastery of enterprise automation, workflow management, and business process integration through sophisticated platform development
- Build portfolio showcase of advanced automation capabilities including workflow engines, monitoring, and enterprise deployment
- Develop systematic approach to automation design, implementation, and management for complex business environments
- Create reusable frameworks and methodologies for enterprise-level automation and workflow management
- Establish foundation for advanced roles in automation engineering, business process management, and enterprise technology leadership
- Show integration of technical automation skills with business requirements, compliance, and enterprise software development
- Contribute to automation field advancement through demonstrated mastery of fundamental automation concepts applied to complex business environments

---

_Your mastery of automation represents a crucial milestone in practical software development. Automation skills transform you from someone who can write code to someone who can solve real business problems and improve operational efficiency at scale. The systematic thinking, reliability focus, and business process understanding you develop through automation will serve as the foundation for roles in operations, data engineering, business analysis, and enterprise software development. Each automation system you build teaches you not just technical skills, but also business process thinking and the responsibility that comes with building systems that impact real workflows and business operations._
