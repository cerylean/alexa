#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written as part of https://www.scrapehero.com/how-to-scrape-amazon-product-reviews-using-python/
from lxml import html
import json
import requests
import pandas as pd
from pandas.io.json import json_normalize
import json,re
from dateutil import parser as dateparser
from time import sleep

def ParseReviews(asin,i):
	# for i in range(5):
	# 	try:
	#This script has only been tested with Amazon.com
	amazon_url  = 'https://www.amazon.com/Echo-Dot-2nd-Generation-speaker/product-reviews/B015TJD0Y4/ref=cm_cr_arp_d_paging_btm_'+str(i)+'?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(i)
	#amazon_url  = 'http://www.amazon.com/All-New-Amazon-Echo-Dot-Add-Alexa-To-Any-Room/product-reviews/B01DFKC2SO/ref=cm_cr_getr_d_paging_btm_'+str(i)+'?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(i)
	# Add some recent user agent to prevent amazon from blocking the request
	# Find some chrome user agent strings  here https://udger.com/resources/ua-list/browser-detail?browser=Chrome
	headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
	page = requests.get(amazon_url,headers = headers,verify=False)
	page_response = page.text

	parser = html.fromstring(page_response)
	XPATH_AGGREGATE = '//span[@id="acrCustomerReviewText"]'
	XPATH_REVIEW_SECTION_1 = '//div[contains(@id,"reviews-summary")]'
	XPATH_REVIEW_SECTION_2 = '//div[@data-hook="review"]'

	XPATH_AGGREGATE_RATING = '//table[@id="histogramTable"]//tr'
	XPATH_PRODUCT_NAME = '//h1//span[@id="productTitle"]//text()'
	XPATH_PRODUCT_PRICE  = '//span[@id="priceblock_ourprice"]/text()'

	raw_product_price = parser.xpath(XPATH_PRODUCT_PRICE)
	product_price = ''.join(raw_product_price).replace(',','')

	raw_product_name = parser.xpath(XPATH_PRODUCT_NAME)
	product_name = ''.join(raw_product_name).strip()
	total_ratings  = parser.xpath(XPATH_AGGREGATE_RATING)
	reviews = parser.xpath(XPATH_REVIEW_SECTION_1)
	if not reviews:
		reviews = parser.xpath(XPATH_REVIEW_SECTION_2)
	ratings_dict = {}
	reviews_list = []

	if not reviews:
		raise ValueError('unable to find reviews in page')

	#grabing the rating  section in product page
	for ratings in total_ratings:
		extracted_rating = ratings.xpath('./td//a//text()')
		if extracted_rating:
			rating_key = extracted_rating[0]
			raw_raing_value = extracted_rating[1]
			rating_value = raw_raing_value
			if rating_key:
				ratings_dict.update({rating_key:rating_value})

	#Parsing individual reviews
	for review in reviews:
		XPATH_RATING  = './/i[@data-hook="review-star-rating"]//text()'
		XPATH_REVIEW_HEADER = './/a[@data-hook="review-title"]//text()'
		XPATH_REVIEW_POSTED_DATE = './/span[@data-hook="review-date"]//text()'
		XPATH_REVIEW_TEXT_1 = './/span[@data-hook="review-body"]//text()'
		XPATH_REVIEW_TEXT_2 = './/div//span[@data-action="columnbalancing-showfullreview"]/@data-columnbalancing-showfullreview'
		XPATH_REVIEW_COMMENTS = './/span[@data-hook="helpful-vote-statement"]//text()'
		XPATH_AUTHOR  = './/a[@data-hook="format-strip"]//text()'
		XPATH_REVIEW_TEXT_3  = './/div[contains(@id,"dpReviews")]/div/text()'

		raw_review_author = review.xpath(XPATH_AUTHOR)
		raw_review_rating = review.xpath(XPATH_RATING)
		raw_review_header = review.xpath(XPATH_REVIEW_HEADER)
		raw_review_posted_date = review.xpath(XPATH_REVIEW_POSTED_DATE)
		raw_review_text1 = review.xpath(XPATH_REVIEW_TEXT_1)
		raw_review_text2 = review.xpath(XPATH_REVIEW_TEXT_2)
		raw_review_text3 = review.xpath(XPATH_REVIEW_TEXT_3)

		#cleaning data
		author = ' '.join(' '.join(raw_review_author).split())
		review_rating = ''.join(raw_review_rating).replace('out of 5 stars','')
		review_header = ' '.join(' '.join(raw_review_header).split())

		try:
			review_posted_date = dateparser.parse(''.join(raw_review_posted_date)).strftime('%d %b %Y')
		except:
			review_posted_date = None
		review_text = ' '.join(' '.join(raw_review_text1).split())

		#grabbing hidden comments if present
		if raw_review_text2:
			json_loaded_review_data = json.loads(raw_review_text2[0])
			json_loaded_review_data_text = json_loaded_review_data['rest']
			cleaned_json_loaded_review_data_text = re.sub('<.*?>','',json_loaded_review_data_text)
			full_review_text = review_text+cleaned_json_loaded_review_data_text
		else:
			full_review_text = review_text
		if not raw_review_text1:
			full_review_text = ' '.join(' '.join(raw_review_text3).split())

		raw_review_comments = review.xpath(XPATH_REVIEW_COMMENTS)
		review_comments = ''.join(raw_review_comments)
		review_comments = re.sub('[A-Za-z]','',review_comments).strip()
		review_dict = {
							'review_comment_count':review_comments,
							'review_text':full_review_text,
							'review_posted_date':review_posted_date,
							'review_header':review_header,
							'review_rating':review_rating,
							'review_author':author

						}
		reviews_list.append(review_dict)

	data = {
				'ratings':ratings_dict,
				'reviews':reviews_list,
				'url':amazon_url,
				'price':product_price,
				'name':product_name
			}
	return data
	# 	except ValueError:
	# 		print("Retrying to get the correct response")

	# return {"error":"failed to process the page","asin":asin}
def parse_json(json_file):
    with open (json_file) as data_file:
        data= json.load(data_file)
        df = pd.io.json.json_normalize(data)
    return df

def parse_reviews(df,col):
    new_df = pd.DataFrame(columns = list(df[col][0][0].keys()))
    for i in range(df.shape[0]):
        for review in df[col][i]:
            new_df = new_df.append(review,ignore_index=True)
    return new_df

def ReadAsin():
	#Add your own ASINs here
	AsinList = ['B01DFKC2SO']
	extracted_data = []
	for i in range(1400,1480):
		print('Downloading and processing page https://www.amazon.com/Echo-Dot-2nd-Generation-speaker/product-reviews/B015TJD0Y4/ref=cm_cr_arp_d_paging_btm_'+str(i)+'?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(i))
		extracted_data.append(ParseReviews('B01DFKC2SO',i))
		sleep(7)
	f = open('data.json','w')
	json.dump(extracted_data,f,indent=4)


if __name__ == '__main__':
	ReadAsin()
	dfOG = parse_json('data.json')
	df = parse_reviews(dfOG,'reviews')
	with open('amazon_reviews.csv', 'a') as f:
		df.to_csv(f, header=False)
