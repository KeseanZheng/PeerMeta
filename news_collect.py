import numpy as np
import pandas as pd
import datetime as dt
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.proxy import Proxy, ProxyType
import time
import re
import os
import random
from selenium.webdriver.chrome.options import Options


service = Service(executable_path = "C:\Program Files\Google\Chrome\Application\chromedriver.exe") #executable_path="C:\Program Files\Google\Chrome\Application\chrome.exe")
chrome_options = Options()
chrome_options.add_argument("--headless")

def switch_search_window():
    new_window_handle = None
    for handle in driver.window_handles:
        if handle != driver.current_window_handle:
            new_window_handle = handle
            break
    if new_window_handle:
        driver.switch_to.window(new_window_handle)
    else:
        raise ValueError("Cannot switch to the new page!")
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'iw')))
    except:
        driver.refresh()
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'iw')))
    zhuanye_button = driver.find_element(by=By.CSS_SELECTOR, value='#form0 > table.bbltab11 > tbody > tr > td.bbltabtd5 > a')
    zhuanye_button.click()
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, 'bbbody')))

    zhongguojingji_button = driver.find_element(by=By.CSS_SELECTOR,value='#bankbody > div.bbbody > div.bbbleft.bspec > table:nth-child(4) > tbody > tr:nth-child(2) > td:nth-child(1) > a')
    zhongguojingji_button.click()
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'iw')))

def search_fraud_news(firm_name):
    driver.find_element(By.ID, "iw").clear()
    driver.find_element(By.ID, 'iw').send_keys(firm_name)
    # driver.find_element(By.ID, 'db').send_keys('中国经济新闻数据库')
    driver.find_element(By.ID, 'rl').send_keys('任意字词命中')
    driver.find_element(By.NAME, 'starTime').send_keys('2001-01-01')
    driver.find_element(By.NAME, 'endTime').send_keys('2023-12-31')
    seach_button = driver.find_element(by=By.CLASS_NAME, value='btn')
    seach_button.click()
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'iw')))

    driver.find_element(By.ID, "iw").clear()
    driver.find_element(By.ID, 'iw').send_keys(FS_terms)
    seach_button = driver.find_element(by=By.CLASS_NAME, value='jgsearch')
    seach_button.click()
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, 'iw')))

    driver.find_element(By.ID, "iw").clear()
    driver.find_element(By.ID, 'iw').send_keys(fraud_terms)
    seach_button = driver.find_element(by=By.CLASS_NAME, value='jgsearch')
    seach_button.click()

def get_news_info(dates, presses, titles, contents):
    i = 0

    o_dates = driver.find_elements(by=By.CSS_SELECTOR, value='td[style="padding-top:7px;color:#206093;width:100px"]')
    o_presses = driver.find_elements(by=By.CSS_SELECTOR, value='td[width="120"]')
    o_titles = driver.find_elements(by=By.CLASS_NAME, value='tabListA')

    for o_date in o_dates:
        dates.append(o_date.text.strip())
    for o_press in o_presses:
        i += 1
        if (i % 2) != 0:
            presses.append(o_press.text.strip())
    for o_title in o_titles:
        titles.append(o_title.text.strip())
        href = o_title.get_attribute('href')
        driver.execute_script("return" + href)
        windows = driver.window_handles
        driver.switch_to.window(windows[-1])
        try:
            page_content = driver.find_element(by=By.ID, value='text').text.strip()
            # WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'text')))
        except:
            try:
                driver.refresh()
                # WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'text')))
                page_content = driver.find_element(by=By.ID, value='text').text.strip()
            except:
                print("The news content cannot be found!")
                exit()
        contents.append(page_content)
        driver.close()
        windows = driver.window_handles
        driver.switch_to.window(windows[-1])

    return dates, presses, titles, contents

def crawled_stock_check():
    news_folder = r"D:\data\firms_fraud_news"
    stock_ls = []

    for filename in os.listdir(news_folder):
        if filename.endswith('.csv'):
            stock_name, _ = os.path.splitext(filename)
            stock_ls.append(stock_name)

    df = pd.read_csv('..\.\stockname_2001-2022.csv')
    df['Symbol'] = df['Symbol'].astype(str)
    df.insert(1, 'Exist', "")
    df['Exist'] = df['Symbol'].apply(lambda x: 1 if x in stock_ls else 0)
    df.to_csv('crawled_stock_check.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    url = r'http://www.bjinfobank.com/indexShow.do?method=index'

    FS_terms = '财务 会计 审计 税款 账目 资产 负债 收入 利润 业绩 偿付 现金流 预算 收支 经营 债务 资金 贷款 费用 融资 投资 报表 财报 披露'
    fraud_terms = '造假 作假 涉嫌 指控 失实 爆出 虚假 弄虚作假 捏造 炒作 不实 隐瞒 利益输送 爆料 起底 查出 不属实 篡改 黑幕 舞弊 欺诈 疑点 违反 虚报 揭露 纠纷 举报 偷工减料 虚增 虚减'

    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(url)
    switch_search_window()

    crawled_stock_check()
    firm_names = pd.read_csv(r"D:\python_work\FSfraud\paper_codes\crawled_stock_check.csv")
    firm_names['Symbol'] = firm_names['Symbol'].astype(str)
    firm_names = firm_names[firm_names['Exist'] == 0].reset_index(drop=True)
    stock_num=len(firm_names['Symbol'])
    stocknames_dict={}
    for i in range(stock_num):
        stocknames=list(set(firm_names.iloc[i,2:].tolist()))
        try:
            stocknames.remove(np.NaN)
        except:
            pass
        stocknames_dict[firm_names['Symbol'][i]]=stocknames

    for code in firm_names['Symbol']:
        firm_name=" ".join(stocknames_dict[code])
        if firm_name == "":
            continue
        print(firm_name)

        search_fraud_news(firm_name)

        # WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, 'tabListTd1')))
        doc_num = int(driver.find_element(by=By.CLASS_NAME, value='mlgreen').text.strip().replace(",", ""))


        if doc_num == 0 or doc_num > 1000:
            print('None or Surplus Warning: There are ' + '{:.0f}'.format(doc_num) + ' results for ' + str(code))
            df = pd.DataFrame(columns=['date', 'press', 'title', 'content'])
            df.to_csv(r'D:\data\firms_fraud_news\\' + code + '.csv', index=False, encoding='utf_8_sig')
        else:
            print('There are ' + '{:.0f}'.format(doc_num) + ' results for ' + str(code))
            max_page = int(doc_num/25) + (0 if doc_num%25 == 0 else 1)

            add_page = False
            if add_page == True:
                add_page_from = 10
                nextpage_button = driver.find_element(by=By.CSS_SELECTOR, value='[title="下一页"]')
                nextpage_button.click()
                current_url = driver.current_url
                skip_url = re.sub(r'page=(\d+)&', f'page={add_page_from}&', current_url)
                driver.get(skip_url)
            else:
                add_page_from = 1

            dates, presses, titles, contents = [], [], [], []
            for n in range(add_page_from, max_page+1):
                if n == 1:
                    dates, presses, titles, contents = get_news_info(dates, presses, titles, contents)
                    if max_page > 1:
                        page2_button = driver.find_element(by=By.CSS_SELECTOR, value='#bankbody > div.bbbody > form > div > div.fenye > table > tbody > tr > td > table > tbody > tr > td:nth-child(2) > a:nth-child(2)')
                        page2_button.click()
                    else:
                        break
                else:
                    dates, presses, titles, contents = get_news_info(dates, presses, titles, contents)
                    current_url = driver.current_url
                    next_url = re.sub(r'page=(\d+)&', f'page={n+1}&', current_url)
                    if n < max_page:
                        driver.get(next_url)
                    else:
                        print('Already reaching the end page!')

                news_data = []
                for (date, press, title, content) in zip(dates, presses, titles, contents):
                    date = pd.to_datetime(date)
                    news_data.append((date, press, title, content))
                df = pd.DataFrame(news_data, columns=['date', 'press', 'title', 'content'])

                df.to_csv(r'D:\data\firms_fraud_news\\' + code + '.csv',index=False,encoding='utf_8_sig')

            news_data = []
            for (date, press, title, content) in zip(dates, presses, titles, contents):
                date = pd.to_datetime(date)
                news_data.append((date, press, title, content))
            df = pd.DataFrame(news_data, columns=['date', 'press', 'title', 'content'])

            df.to_csv(r'D:\data\firms_fraud_news\\' + code + '.csv', index=False, encoding='utf_8_sig')
            print('\rThe fraud news has been collected for -{}-'.format(code))

        zhuanye_button = driver.find_element(by=By.CSS_SELECTOR, value='#bankbody > div.bbbody > form > div > div.bblsearch > table.bbltab11 > tbody > tr > td.bbltabtd5 > a')
        zhuanye_button.click()
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, 'bbbody')))

        zhongguojingji_button = driver.find_element(by=By.CSS_SELECTOR, value='#bankbody > div.bbbody > div.bbbleft.bspec > table:nth-child(4) > tbody > tr:nth-child(2) > td:nth-child(1) > a')
        zhongguojingji_button.click()
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'iw')))
