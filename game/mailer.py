# !/usr/bin/python3
# coding: utf-8

# Copyright YYYY AUTHORS
#
# YOUR COPYRIGHT HERE (IF YOU WANT IT)


""" Send emails """

import base64
import locale
import os
from email.mime.text import MIMEText

from game.gmail import GMailApiOAuth, send_email

# script settings

THIS_FOLDER = os.path.dirname(os.path.realpath(__file__))
OAUTH_FOLDER = os.path.join(THIS_FOLDER, ".user_credentials", "gmail")

# email settings
APP_NAME = "GAME | Your results"
EMAIL_DRIVER = GMailApiOAuth(
    "GAME",
    os.path.join(OAUTH_FOLDER, "client_secret.json"),
    os.path.join(OAUTH_FOLDER, "gmail.json")
).create_driver()
EMAIL_SENDER = "game.cosmosns@gmail.com"

# setting locale
locale.setlocale(locale.LC_ALL, "it_IT.UTF-8")  # italian


def get_msg():
    """
    :return: MIMEText
        Personalized message to notify user
    """

    message = MIMEText(
        "<html>" +
        "header\n" +
        "content\n" +
        "footer\n" +
        "</html>", "html"
    )
    message["subject"] = "subject"
    message["to"] = "sirfoga@protonmail.com"

    return {
        "raw": base64.urlsafe_b64encode(bytes(message)).decode()
    }


def send_msg(msg):
    """
    :param msg: str
        Message to send to me
    :return: void
        Sends email to me with this message
    """

    send_email(
        EMAIL_SENDER,
        msg,
        EMAIL_DRIVER
    )


if __name__ == '__main__':
    send_msg(get_msg())
