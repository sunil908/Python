# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:42:20 2018

@author: GEFA0728
"""

from __future__ import print_function

__author__ = '"speech2text-201415@speech2text-201415.iam.gserviceaccount.com"'

from googleapiclient.discovery import build


def main():

  # Build a service object for interacting with the API. Visit
  # the Google APIs Console <http://code.google.com/apis/console>
  # to get an API key for your own application.
  service = build('translate', 'v2',
            developerKey='AIzaSyDRRpR3GS1F1_jKNNM9HCNd2wJQyPG3oN0')
  print(service.translations().list(
      source='en',
      target='fr',
      q=['flower', 'car']
    ).execute())

if __name__ == '__main__':
  main()