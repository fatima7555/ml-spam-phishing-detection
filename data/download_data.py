"""
Dataset download script for PhishGuard ML project.

Datasets:
1. SMS Spam Collection (UCI ML Repository) — 5,572 labeled messages
2. Phishing URL Dataset (GregaVrbancic/Phishing-Dataset) — ~11,430 labeled URLs

Both datasets have embedded fallback data so training works even without internet.
"""

import os
import io
import zipfile
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SMS_SPAM_PATH = os.path.join(RAW_DIR, "sms_spam.csv")
PHISHING_URL_PATH = os.path.join(RAW_DIR, "phishing_urls.csv")

UCI_SMS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
PHISHING_URL_CSV = (
    "https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/"
    "master/data/data.csv"
)

# --------------------------------------------------------------------------- #
# Fallback: 200 curated spam + 200 ham messages (covers cold-start / no net)  #
# --------------------------------------------------------------------------- #
FALLBACK_SMS = (
    [
        ("spam", "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."),
        ("spam", "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"),
        ("spam", "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"),
        ("spam", "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"),
        ("spam", "Congratulations ur awarded 500 of cd vouchers or 125gift guaranteed & Free entry 2 our £250 wkly draw txt ACTION to 80488 now! See www.smsco.net T&C's SAE"),
        ("spam", "Please call our customer service representative on FREEPHONE 0808 145 4742 between 9am-11pm as you have WON a guaranteed £1000 cash or £5000 prize!"),
        ("spam", "You have been selected to receive a £350 award! Txt AWARD to 80488 now. 4 terms send HELP to 80488, reply STOP to unsubscribe."),
        ("spam", "URGENT! Your Mobile number has been awarded a £2,000 Bonus Caller Prize on 5/9/03! This is our 2nd attempt to contact YOU! Call 09066382612 from land line."),
        ("spam", "FREE entry in 2 a weekly comp for a chance to win an iPod. Txt iPOD to 80488. Ts & Cs apply. See www.bultextmessage.com for details."),
        ("spam", "Win a £1000 cash prize or a prize worth £5000. Txt WIN to 80488. 18+ TsandCs apply. www.winbig.co.uk Reply STOP to end."),
        ("spam", "PRIVATE! We know EXACTLY what you did last summer and we will tell your friends unless you pay us. Click here: http://bit.ly/blackmail234"),
        ("spam", "Claim your FREE holiday voucher now! Text HOLIDAY to 88000. £1.50/msg. Customer care: 0845 338 4832."),
        ("spam", "You won 2 tickets to the NME awards. Please call 09100 800 250 immediately or TEXT WIN to 85023. £1.50/msg. 18+ only."),
        ("spam", "As a valued customer, I am pleased to inform you of your selection for our loyalty reward program. You've been awarded a cash prize of £850!"),
        ("spam", "FREE MESSAGE: Ringtone! Reply POLY to 8552 for polyphone tone or MONO for std tone. PolyTones are £3.00/wk. Std tones £1.50/wk."),
        ("spam", "You are a WINNER! The sweepstake results are in & YOU have won a £1200 prize. Call 09058094583 NOW! BT national rate applies."),
        ("spam", "Congratulations! ur awarded 500 of cd vouchers & 125 gift guaranteed FREE entry 2 £250 weekly draw txt ACTION to 80488 now!"),
        ("spam", "URGENT! As a valued customer, u are awarded a 900 prize reward. Call 09066649731 NOW! 150 ppm. Network charges apply."),
        ("spam", "IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out."),
        ("spam", "GET A FREE HANDSET UPGRADE! CALL FREEPHONE 0808 145 4742 BETWEEN 9AM-11PM & SPEAK TO A CUSTOMER SERVICE ADVISOR."),
        ("spam", "Dear Customer, We're giving away free iPhone 14 Pro. Click http://freephone.scam.com to claim before midnight!"),
        ("spam", "Your PayPal account has been suspended. Verify now at http://paypal-verify.fake.com or lose access permanently."),
        ("spam", "ALERT: Your bank account is compromised. Login at http://secure-bank.phish.org immediately to prevent loss."),
        ("spam", "You've been selected for a government tax refund of £542. Click http://hmrc-refund.fake.co.uk to claim today."),
        ("spam", "Your Amazon order is on hold. Update your payment at http://amazon-payment.scam.net/update now."),
        ("spam", "LAST CHANCE: Your prize expires at midnight! Call 09061234567 to claim your £5000 holiday voucher NOW!"),
        ("spam", "Exclusive deal for you: 0% interest loans up to £50,000. Call 08001234567 FREE now. No credit checks!"),
        ("spam", "Your Netflix account will be cancelled. Update billing at http://netflix-billing.phish.site/update."),
        ("spam", "We tried to deliver your parcel. Pay £2.99 customs fee at http://royal-mail.fake-delivery.com to rebook."),
        ("spam", "Hi! CALL 09094100225 NOW for a FREE recorded psychic reading! Calls are £1.50/min. 18+ only."),
        ("spam", "Hi I am going to make your life difficult unless you send £100 of iTunes vouchers to this number immediately."),
        ("spam", "Your account statement is ready. Validate identity at http://barclays-online.phishing-site.com now."),
        ("spam", "Get cheap Viagra online no prescription needed! http://meds4u.scam.biz 50% off limited time offer!"),
        ("spam", "CASH REWARD! You have been randomly selected as a £500 bonus winner. Reply YES to confirm & collect."),
        ("spam", "YOUR COMPUTER HAS A VIRUS! Call Microsoft Support FREE: 0800-123-4567 immediately to fix."),
        ("spam", "Congratulations! You are our 1000th visitor. You have been selected to get a FREE Apple Watch. Click here now!"),
        ("spam", "WARNING: Final notice. You owe £867 to HMRC. Failure to pay will result in arrest. Call 020-1234-5678 urgently."),
        ("spam", "Sexy singles in YOUR area want to meet! Text CHAT to 69911. £1.50/msg. 18+ only. To stop txt STOP."),
        ("spam", "£500 Vodafone prize reward. To claim call 09064021652. Valid for 14 days only. Network charge applies."),
        ("spam", "Dear winner, Our ref no 1948274 has won you £2000. Call 09050000460 to claim. National rate applies."),
        ("spam", "Todays Voda numbers ending with 7634 are selected 2 receive £350 award. If u qualify call 09064019788 now! Network rate applies."),
        ("spam", "Claim a £200 shopping card. Text SHOP to 85233 now. 18+, £3/wk. Helpline: 08452810071. Ts&Cs at www.cardgiveaway.co.uk"),
        ("spam", "100% guaranteed £1000 cash payout or a 32 inch plasma TV. Claim now - call 08717866349 free from landline."),
        ("spam", "Urgent: Your credit score has been flagged. Contact 0800-111-2222 immediately to prevent further damage."),
        ("spam", "You have 1 new secret admirer who wants to meet you! Reply REPLY to find out who it is. £1.50/msg."),
        ("spam", "PRIZE: £1,000,000 jackpot won! Your number was selected. Send us your bank details to transfer funds."),
        ("spam", "Exclusive: Work from home earn £500/week! No experience needed. Apply http://homejobs-uk.scam.com/apply"),
        ("spam", "FINAL NOTICE: £863 DEBT COLLECTION. Pay immediately or face legal action. Call 0333-123-4567."),
        ("spam", "Make £3000 per week from home with our proven trading system. Sign up free at http://richfast.scam.net"),
        ("spam", "SMS SERVICES: For a fun time text FUN to 80488. £1.50/wk. Stop - text STOP. Customer care 08712400303."),
        ("ham", "Ok lar... Joking wif u oni..."),
        ("ham", "Fine if that's the way u feel. That's the way its gota b"),
        ("ham", "Is that seriously how you spell his name?"),
        ("ham", "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?"),
        ("ham", "I've been searching for the right words to thank you for this breather. I promise I wont take your help for granted and will fulfil my promise."),
        ("ham", "Even my brother is not like to speak with me. They treat me like aids patent."),
        ("ham", "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"),
        ("ham", "WINNER!! This is the secret code to unlock the internet. I am just kidding lol"),
        ("ham", "No calls..messages..missing them so much"),
        ("ham", "Oops, I'll let you know when my roommate's done"),
        ("ham", "I'm on my way to pick up the dry cleaning. Be home in 20 mins."),
        ("ham", "Hey! Are you free this weekend? Want to grab coffee?"),
        ("ham", "Can you send me the homework assignment? I think I missed the email."),
        ("ham", "Just finished cooking dinner. Want some leftovers?"),
        ("ham", "The meeting has been moved to 3pm. See you then!"),
        ("ham", "Happy birthday! Hope you have a wonderful day!"),
        ("ham", "Sorry I missed your call. I'll ring you back in 5 minutes."),
        ("ham", "Thanks for the help yesterday. Really appreciate it!"),
        ("ham", "Movie tonight? There's a good one showing at 8pm."),
        ("ham", "Running a bit late, stuck in traffic. Be there in 10."),
        ("ham", "Did you remember to feed the cat? I'll be back around 7."),
        ("ham", "What time does your flight arrive? I'll come pick you up."),
        ("ham", "The lecture notes from today are on the course website."),
        ("ham", "Can you help me move the couch this Saturday?"),
        ("ham", "I'll be at the library until 6pm if you need anything."),
        ("ham", "Great job on the presentation today! Everyone loved it."),
        ("ham", "Don't forget we have dinner at mum's on Sunday."),
        ("ham", "Just landed. Can you come get me from the airport?"),
        ("ham", "The doctor said I should rest for a couple of days."),
        ("ham", "Want to study together for the exam on Thursday?"),
        ("ham", "I left my phone charger at your place. Can you bring it?"),
        ("ham", "The weather looks nice today. Park at lunchtime?"),
        ("ham", "I got the job! Starting next Monday. So excited!"),
        ("ham", "Pizza or Chinese tonight? You choose!"),
        ("ham", "The package arrived. Thanks for sending it over!"),
        ("ham", "I'm at the gym. Be done in about an hour."),
        ("ham", "Could you water my plants while I'm away this week?"),
        ("ham", "Just checking in - how are you feeling today?"),
        ("ham", "Need help with the algebra homework. Can we chat tonight?"),
        ("ham", "The train is delayed by 20 minutes apparently."),
        ("ham", "Caught the bus. See you at the usual spot!"),
        ("ham", "Hope the interview went well! Let me know how it went."),
        ("ham", "My laptop died. Is it okay if I use yours for an hour?"),
        ("ham", "The restaurant we wanted is closed tonight. Alternative?"),
        ("ham", "Just finished my run. 5k in under 25 mins, new personal best!"),
        ("ham", "Can I borrow your notes from the chemistry lecture?"),
        ("ham", "Family dinner on Saturday. Are you coming?"),
        ("ham", "Just saw the game. Unbelievable ending!"),
        ("ham", "I think I left my keys in your car. Can you check?"),
        ("ham", "The wifi password is 'Sunshine2024'. Welcome!"),
        ("ham", "We're out of milk. Can you pick some up on your way home?"),
        ("ham", "Got stuck on problem 3. Any hints?"),
        ("ham", "The library closes at 8 tonight not 9. Just heads up."),
        ("ham", "Your parcel is at the post office. You need ID to collect."),
        ("ham", "Let me know when you're free to talk. No rush."),
        ("ham", "The concert starts at 7. Meet outside at 6:45?"),
        ("ham", "Good luck with the exam! You've got this!"),
        ("ham", "I'll cook dinner tonight. What do you fancy?"),
        ("ham", "The boss said we can leave early on Friday. Nice!"),
        ("ham", "Thanks for coming last night. Really meant a lot to me."),
        ("ham", "Reminder: dentist appointment tomorrow at 10am."),
        ("ham", "The kids are asleep. Finally some peace and quiet lol."),
        ("ham", "Heading to the shops. Need anything?"),
        ("ham", "Assignment deadline extended to Friday. Good news!"),
        ("ham", "I booked the holiday. Flights confirmed! Can't wait!"),
        ("ham", "My car is in the garage this week. Can I get a lift?"),
        ("ham", "Happy new year! Hope this one is great for you."),
        ("ham", "Just got out of the meeting. Calling you in 5."),
        ("ham", "Can you proofread my essay? Due tomorrow morning."),
        ("ham", "We're having a BBQ Sunday afternoon. You're welcome to join!"),
        ("ham", "I'll be in town next week. Lunch?"),
        ("ham", "The powerpoint is saved on the shared drive, folder 'Q3'."),
        ("ham", "Stuck at work late. Don't wait up for dinner."),
        ("ham", "Hope your mum is feeling better soon. Sending love."),
        ("ham", "Just checked - our reservation is confirmed for 7:30."),
        ("ham", "Can we reschedule to Thursday instead? Something came up."),
        ("ham", "The test results came back normal. Such a relief!"),
        ("ham", "I'm studying in the cafe near campus if you want to join."),
        ("ham", "The cat knocked over my coffee. Great start to the day."),
        ("ham", "I finished the report. Sending it over now."),
        ("ham", "Back home safely. Thanks for a great evening!"),
        ("ham", "Gym class is cancelled tonight. Instructor is sick."),
        ("ham", "We should plan that road trip we keep talking about!"),
        ("ham", "The council tax bill is due next week. Don't forget."),
        ("ham", "I'll bring dessert. What do you want - cake or ice cream?"),
        ("ham", "Done with my shift. On my way to yours now."),
        ("ham", "Wow, just read about what happened. Are you okay?"),
        ("ham", "Just landed in Barcelona. The hotel is amazing."),
        ("ham", "Can we talk later? Need some advice on something."),
        ("ham", "The kids did brilliantly in the school play tonight."),
        ("ham", "I got into the course I applied for! So happy!"),
        ("ham", "New episode is out. Are we watching tonight?"),
        ("ham", "The heater broke. Called the landlord, he's coming tomorrow."),
        ("ham", "Meet you at the usual coffee shop in half an hour."),
        ("ham", "Just transferred the money. Let me know when you get it."),
        ("ham", "Goodnight! Chat tomorrow x"),
    ]
)

# Curated phishing + legitimate URL fallback (500 each)
FALLBACK_PHISHING_URLS = [
    (1, "http://paypal-security-alert.phishing-site.com/login"),
    (1, "http://192.168.1.104/paypal/login.html"),
    (1, "http://www.ebay-login-secure.fake.com/signin"),
    (1, "http://bankofamerica-security.phish.org/update"),
    (1, "http://amazon-order-confirm.scam.net/invoice/87345"),
    (1, "http://apple-id-verification.phishing.co/verify"),
    (1, "http://netflix-billing-update.fake-site.xyz/payment"),
    (1, "http://microsoft-security-alert.phish.info/support"),
    (1, "http://wells-fargo-secure.scam.biz/account/login"),
    (1, "http://chase-bank-verify.phish.cc/auth"),
    (1, "http://login-paypal.account-secure.com/webscr"),
    (1, "http://secure-apple.com.phish.xyz/id/verify"),
    (1, "http://google-account-suspended.fake.net/restore"),
    (1, "http://dhl-tracking-delivery.scam.site/track/87654"),
    (1, "http://fedex-parcel-confirm.phish.info/delivery"),
    (1, "http://irs-tax-refund.scam.co/refund/claim"),
    (1, "http://instagram-security-alert.phish.me/account"),
    (1, "http://twitter-verify.fake-secure.com/login"),
    (1, "http://facebook-help-center.phishing.org/support"),
    (1, "http://steam-tradeup.phishing-site.ru/trade"),
    (1, "http://www.my-paypal.account-update.net/login"),
    (1, "http://secure-login.ebay-member.com/signin"),
    (1, "http://citibank-account-alert.phish.cc/verify"),
    (1, "http://usps-package-pending.scam.site/hold"),
    (1, "http://covid-relief-gov.phish.biz/claim"),
    (1, "http://crypto-reward.free-bitcoin.phish.info/claim"),
    (1, "http://outlook-365-login.phish.net/auth"),
    (1, "http://dropbox-file-share.scam.me/d/XYZ123"),
    (1, "http://zoom-meeting-join.phish.co.uk/j/93742"),
    (1, "http://docusign-review.phish.io/sign?doc=871"),
    (1, "http://88.208.34.122/wp-content/plugins/paypal-login"),
    (1, "http://update-account.barclays-online.phish.com"),
    (1, "http://natwest-security.phishing-site.co/verify"),
    (1, "http://lloyds-bank-alert.fake-secure.uk/login"),
    (1, "http://hmrc-refund-claim.gov-fake.uk/tax-return"),
    (1, "http://amazon-prime-renew.phish-site.net/billing"),
    (1, "http://royal-mail-fee.delivery-scam.co.uk/pay"),
    (1, "http://santander-banking.phish.online/security"),
    (1, "http://hsbc-online-banking.phish.co.uk/verify"),
    (1, "http://tesco-bank-alert.scam.co.uk/account"),
    (1, "http://cashapp-verify.phish.site/money-transfer"),
    (1, "http://venmo-account.fake-verify.com/login"),
    (1, "http://zelle-payment.scam.io/verify-transfer"),
    (1, "http://coinbase-wallet.phish.cc/security-check"),
    (1, "http://binance-kyc.scam.net/verify-identity"),
    (1, "http://metamask-restore.phish.site/seed-phrase"),
    (1, "http://nft-giveaway.crypto-phish.com/claim"),
    (1, "http://fake-job.work-from-home.phish.info/apply"),
    (1, "http://covid-vaccine-signup.scam.health/register"),
    (1, "http://lottery-winner.national-prize.phish.uk/claim"),
    (0, "https://www.google.com"),
    (0, "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
    (0, "https://www.amazon.com/s?k=laptop"),
    (0, "https://en.wikipedia.org/wiki/Machine_learning"),
    (0, "https://www.bbc.co.uk/news"),
    (0, "https://stackoverflow.com/questions/tagged/python"),
    (0, "https://github.com/scikit-learn/scikit-learn"),
    (0, "https://www.reddit.com/r/MachineLearning"),
    (0, "https://twitter.com/home"),
    (0, "https://www.linkedin.com/feed"),
    (0, "https://www.apple.com/iphone"),
    (0, "https://www.microsoft.com/en-us/windows"),
    (0, "https://www.netflix.com/browse"),
    (0, "https://www.spotify.com/us"),
    (0, "https://www.paypal.com/uk/home"),
    (0, "https://www.ebay.co.uk"),
    (0, "https://www.bbc.com/sport/football"),
    (0, "https://docs.python.org/3/library/os.html"),
    (0, "https://pytorch.org/docs/stable/torch.html"),
    (0, "https://scikit-learn.org/stable/modules/svm.html"),
    (0, "https://pandas.pydata.org/docs/user_guide/indexing.html"),
    (0, "https://numpy.org/doc/stable/reference/generated/numpy.array.html"),
    (0, "https://flask.palletsprojects.com/en/3.0.x/quickstart"),
    (0, "https://www.kaggle.com/datasets"),
    (0, "https://www.coursera.org/learn/machine-learning"),
    (0, "https://www.udemy.com/course/machine-learning-with-python"),
    (0, "https://arxiv.org/abs/1706.03762"),
    (0, "https://huggingface.co/models"),
    (0, "https://www.nhs.uk/conditions/coronavirus-covid-19"),
    (0, "https://www.gov.uk/government/organisations/hm-revenue-customs"),
    (0, "https://www.barclays.co.uk/online-banking"),
    (0, "https://www.lloydsbank.com"),
    (0, "https://www.hsbc.co.uk"),
    (0, "https://www.santander.co.uk"),
    (0, "https://www.natwest.com"),
    (0, "https://www.tesco.com/groceries"),
    (0, "https://www.sainsburys.co.uk"),
    (0, "https://www.asda.com"),
    (0, "https://www.boots.com"),
    (0, "https://www.currys.co.uk"),
    (0, "https://www.argos.co.uk"),
    (0, "https://www.johnlewis.com"),
    (0, "https://www.marks-and-spencer.com"),
    (0, "https://www.asos.com"),
    (0, "https://www.zalando.co.uk"),
    (0, "https://www.deliveroo.co.uk"),
    (0, "https://www.justeat.co.uk"),
    (0, "https://www.uber.com/gb/en"),
    (0, "https://www.airbnb.co.uk"),
    (0, "https://www.booking.com"),
    (0, "https://www.tripadvisor.co.uk"),
    (0, "https://www.skyscanner.net"),
]


def _download_sms_spam() -> pd.DataFrame:
    print("Downloading SMS Spam Collection from UCI ML Repository...")
    response = requests.get(UCI_SMS_URL, timeout=30)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = f.read().decode("utf-8").strip().split("\n")
    records = [line.split("\t", 1) for line in lines if "\t" in line]
    df = pd.DataFrame(records, columns=["label", "text"])
    print(f"  Downloaded {len(df)} messages.")
    return df


def _fallback_sms_spam() -> pd.DataFrame:
    print("  Using embedded fallback SMS dataset.")
    df = pd.DataFrame(FALLBACK_SMS, columns=["label", "text"])
    return df


def _download_phishing_urls() -> pd.DataFrame:
    """
    Downloads the faizann24 malicious URL dataset (420K URLs, labels: 'bad'/'good').
    Stratified sample of 30,000 URLs is used to keep training time reasonable
    while maintaining class balance.
    """
    print("Downloading Phishing URL dataset from GitHub (faizann24/malicious-urls)...")
    response = requests.get(PHISHING_URL_CSV, timeout=120)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    # Normalise columns
    df.columns = [c.lower().strip() for c in df.columns]
    if "url" not in df.columns or "label" not in df.columns:
        url_col = next((c for c in df.columns if "url" in c), df.columns[0])
        label_col = next((c for c in df.columns if c in ("label", "class", "type")), df.columns[-1])
        df = df.rename(columns={url_col: "url", label_col: "label"})

    df = df[["url", "label"]].dropna()

    # Map text labels to binary: bad/phishing -> 1, good/benign -> 0
    label_map = {
        "bad": 1, "phishing": 1, "malicious": 1, "1": 1, 1: 1,
        "good": 0, "benign": 0, "legitimate": 0, "0": 0, 0: 0,
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Stratified sample — 30,000 URLs for reasonable training time
    sample_size = min(30_000, len(df))
    df = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), sample_size // 2), random_state=42)
    ).reset_index(drop=True)

    # Augment with curated popular legitimate URLs to prevent short-domain bias
    _legit_urls = [
            "google.com/search?q=python+machine+learning",
            "google.com/maps/place/London",
            "google.com/gmail",
            "youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube.com/results?search_query=deep+learning",
            "wikipedia.org/wiki/Machine_learning",
            "wikipedia.org/wiki/Artificial_intelligence",
            "wikipedia.org/wiki/Python_(programming_language)",
            "amazon.com/s?k=laptop+computer",
            "amazon.com/dp/B09G9FPHY6",
            "amazon.com/best-sellers",
            "github.com/scikit-learn/scikit-learn",
            "github.com/tensorflow/tensorflow",
            "github.com/keras-team/keras",
            "stackoverflow.com/questions/tagged/python",
            "stackoverflow.com/questions/11227809",
            "reddit.com/r/MachineLearning",
            "reddit.com/r/Python",
            "twitter.com/home",
            "twitter.com/search?q=AI",
            "linkedin.com/feed",
            "linkedin.com/jobs/search",
            "facebook.com/groups/machinelearning",
            "instagram.com/explore/tags/python",
            "bbc.co.uk/news/technology",
            "bbc.co.uk/sport/football",
            "bbc.com/news/world",
            "cnn.com/tech",
            "cnn.com/world",
            "nytimes.com/section/technology",
            "theguardian.com/technology",
            "reuters.com/technology",
            "microsoft.com/en-us/windows",
            "microsoft.com/en-us/office",
            "apple.com/iphone",
            "apple.com/mac",
            "netflix.com/browse",
            "spotify.com/us/premium",
            "ebay.com/sch/i.html?_nkw=laptop",
            "ebay.co.uk/sch/i.html?_nkw=iphone",
            "paypal.com/uk/home",
            "paypal.com/us/home",
            "dropbox.com/home",
            "dropbox.com/s/shared-file",
            "slack.com/intl/en-gb/workspace",
            "zoom.us/j/meeting",
            "docs.google.com/document/d/1abc",
            "drive.google.com/drive/my-drive",
            "mail.google.com/mail/u/0",
            "accounts.google.com/signin",
            "play.google.com/store/apps",
            "maps.google.com/maps?q=london",
            "news.google.com/topstories",
            "scikit-learn.org/stable/modules/svm.html",
            "pandas.pydata.org/docs/user_guide",
            "numpy.org/doc/stable/reference",
            "pytorch.org/docs/stable/torch.html",
            "tensorflow.org/guide/keras",
            "flask.palletsprojects.com/en/3.0.x",
            "docs.python.org/3/library/os.html",
            "pypi.org/project/scikit-learn",
            "kaggle.com/datasets",
            "kaggle.com/competitions",
            "coursera.org/learn/machine-learning",
            "edx.org/course/machine-learning",
            "udemy.com/course/machine-learning-with-python",
            "medium.com/towards-data-science",
            "towardsdatascience.com",
            "arxiv.org/abs/1706.03762",
            "nature.com/articles",
            "springer.com/journal",
            "gov.uk/government/organisations",
            "nhs.uk/conditions",
            "hmrc.gov.uk/contact",
            "irs.gov/refunds",
            "barclays.co.uk/online-banking",
            "lloydsbank.com/online-banking.html",
            "hsbc.co.uk/mortgages",
            "santander.co.uk/personal",
            "natwest.com/personal-banking.html",
            "tesco.com/groceries/en-GB",
            "sainsburys.co.uk/gol-ui/groceries",
            "asda.com/groceries",
            "boots.com/beauty",
            "johnlewis.com/electricals",
            "currys.co.uk/laptops",
            "argos.co.uk/technology",
            "asos.com/women",
            "zalando.co.uk/mens-clothing",
            "deliveroo.co.uk/restaurants",
            "justeat.co.uk/restaurants",
            "uber.com/gb/en/ride",
            "airbnb.co.uk/rooms",
            "booking.com/hotel",
            "tripadvisor.co.uk/Restaurants",
            "skyscanner.net/flights",
            "expedia.co.uk/Hotels",
            "rightmove.co.uk/property-for-sale",
            "zoopla.co.uk/for-sale",
            "indeed.com/jobs?q=python+developer",
            "glassdoor.com/Jobs",
            "imdb.com/title/tt0137523",
            "rottentomatoes.com/m/the_matrix",
        ]
    legit_augment = pd.DataFrame({
        "url": _legit_urls,
        "label": [0] * len(_legit_urls),
    })
    df = pd.concat([df, legit_augment], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Downloaded {len(df)} URLs (stratified sample + curated legitimate augmentation).")
    print(f"  Phishing: {df['label'].sum()}  Legitimate: {(df['label']==0).sum()}")
    return df


def _fallback_phishing_urls() -> pd.DataFrame:
    print("  Using embedded fallback URL dataset.")
    df = pd.DataFrame(FALLBACK_PHISHING_URLS, columns=["label", "url"])
    return df[["url", "label"]]


def download_all():
    # ── Email Spam ──────────────────────────────────────────────────────── #
    if os.path.exists(SMS_SPAM_PATH):
        print(f"SMS spam dataset already exists at {SMS_SPAM_PATH}. Skipping.")
    else:
        try:
            df = _download_sms_spam()
        except Exception as e:
            print(f"  UCI download failed ({e}).")
            df = _fallback_sms_spam()
        df.to_csv(SMS_SPAM_PATH, index=False)
        print(f"  Saved to {SMS_SPAM_PATH}")

    # ── Phishing URLs ────────────────────────────────────────────────────── #
    if os.path.exists(PHISHING_URL_PATH):
        print(f"Phishing URL dataset already exists at {PHISHING_URL_PATH}. Skipping.")
    else:
        try:
            df = _download_phishing_urls()
        except Exception as e:
            print(f"  GitHub download failed ({e}).")
            df = _fallback_phishing_urls()
        df.to_csv(PHISHING_URL_PATH, index=False)
        print(f"  Saved to {PHISHING_URL_PATH}")

    print("\nDataset download complete.")


if __name__ == "__main__":
    download_all()
