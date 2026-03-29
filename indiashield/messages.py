from typing import Dict, List
import random

REAL_NEWS = [
    {
        "id": "real_001",
        "text": "Dilli mein aaj se odd-even rule lagu. Sarkaari notification check karein.",
        "language": "hinglish",
        "topic": "traffic",
        "is_misinfo": False
    },
    {
        "id": "real_002",
        "text": "RBI ne repo rate 6.5% pe rakha. EMI mein koi badlaav nahi.",
        "language": "hindi",
        "topic": "finance",
        "is_misinfo": False
    },
    {
        "id": "real_003",
        "text": "IPL 2024 schedule released. Check BCCI website for details.",
        "language": "english",
        "topic": "sports",
        "is_misinfo": False
    },
    {
        "id": "real_004",
        "text": "Chennai mein aaj baarish ki sambhaavna. IMD ne yellow alert jaari kiya.",
        "language": "hinglish",
        "topic": "weather",
        "is_misinfo": False
    },
    {
        "id": "real_005",
        "text": "தமிழ்நாட்டில் நாளை பொது விடுமுறை அறிவிக்கப்பட்டது.",
        "language": "tamil",
        "topic": "holiday",
        "is_misinfo": False
    },
]

TASK1_MISINFO = [
    {
        "id": "t1_001",
        "text": "URGENT: Sarkar ne secret order nikala — is saal ALL patakhe completely ban. Ye news channels ko paise deke chup karaya gaya hai. ABHI apne ghar waalon ko bhejo. [Forwarded many times]",
        "language": "hinglish",
        "topic": "diwali",
        "is_misinfo": True,
        "spread_rate": 0.7,
        "trust_multiplier": 2.5
    },
    {
        "id": "t1_002",
        "text": "Confirmed: Is baar Diwali pe Chinese crackers mein hidden camera aur GPS tracker mila. CBI jaanch shuru. Share karo.",
        "language": "hindi",
        "topic": "diwali",
        "is_misinfo": True,
        "spread_rate": 0.6,
        "trust_multiplier": 2.0
    },
    {
        "id": "t1_003",
        "text": "Breaking: Diwali ke din curfew lagega 6 baje se. Police ne WhatsApp pe confirm kiya. Sab ko batao.",
        "language": "hinglish",
        "topic": "diwali",
        "is_misinfo": True,
        "spread_rate": 0.8,
        "trust_multiplier": 3.0
    },
]

TASK2_MISINFO = [
    {
        "id": "t2_001",
        "text": "Proof aa gaya — UP mein EVM hack hua. Ye video dekho aur maximum share karo. Election se pehle sabko dikhao.",
        "language": "hindi",
        "topic": "election",
        "is_misinfo": True,
        "spread_rate": 0.75,
        "trust_multiplier": 2.8
    },
    {
        "id": "t2_002",
        "text": "EVM திருட்டு நடக்கிறது — இதை பார்க்கவும் மற்றும் உங்கள் தொடர்புகளுக்கு பகிரவும். தேர்தல் முடிவுகள் மாற்றப்படுகின்றன.",
        "language": "tamil",
        "topic": "election",
        "is_misinfo": True,
        "spread_rate": 0.7,
        "trust_multiplier": 2.5
    },
    {
        "id": "t2_003",
        "text": "Vote nahi diya toh Aadhaar card cancel ho jayega. Government ne ye rule secretly pass kiya. Forward karo jaldi.",
        "language": "hinglish",
        "topic": "election",
        "is_misinfo": True,
        "spread_rate": 0.85,
        "trust_multiplier": 3.5
    },
    {
        "id": "t2_004",
        "text": "Ye candidate ka asli criminal record — media nahi dikhayega kyunki inhe paise mile hain. Viral karo.",
        "language": "hindi",
        "topic": "election",
        "is_misinfo": True,
        "spread_rate": 0.65,
        "trust_multiplier": 2.2
    },
]

TASK3_MISINFO = [
    {
        "id": "t3_001",
        "text": "KHABARDAR: Sarkari hospitals mein dengue ke naam pe zeher inject kar rahe hain. Mere relative ke saath hua. Kisi ko bhi government hospital mat jaane do. SHARE KARO YE BACHAO.",
        "language": "hinglish",
        "topic": "health",
        "is_misinfo": True,
        "spread_rate": 0.9,
        "trust_multiplier": 4.0
    },
    {
        "id": "t3_002",
        "text": "அரசு மருத்துவமனைகளில் டெங்கு என்று சொல்லி விஷம் செலுத்துகிறார்கள். யாரையும் அரசு மருத்துவமனைக்கு அனுப்பாதீர்கள். உடனே பகிருங்கள்.",
        "language": "tamil",
        "topic": "health",
        "is_misinfo": True,
        "spread_rate": 0.85,
        "trust_multiplier": 3.8
    },
    {
        "id": "t3_003",
        "text": "Ek doctor ne sach bataya — ye dengue vaccine mein actually virus hai. YouTube ne video delete kar diya. Telegram pe hai abhi bhi. Apne bachon ko bachao.",
        "language": "hindi",
        "topic": "health",
        "is_misinfo": True,
        "spread_rate": 0.8,
        "trust_multiplier": 3.5
    },
    {
        "id": "t3_004",
        "text": "సర్కారు ఆసుపత్రుల్లో డెంగీ పేరుతో విషం ఇస్తున్నారు. మీ కుటుంబాన్ని కాపాడుకోండి. ఇప్పుడే షేర్ చేయండి.",
        "language": "telugu",
        "topic": "health",
        "is_misinfo": True,
        "spread_rate": 0.8,
        "trust_multiplier": 3.5
    },
    {
        "id": "t3_005",
        "text": "সরকারি হাসপাতালে ডেঙ্গু বলে বিষ দেওয়া হচ্ছে। কাউকে সরকারি হাসপাতালে যেতে দেবেন না। এখনই শেয়ার করুন।",
        "language": "bengali",
        "topic": "health",
        "is_misinfo": True,
        "spread_rate": 0.8,
        "trust_multiplier": 3.5
    },
]

TASK4_MISINFO = [
    {
        "id": "t4_001",
        "text": "CONFIRMED: Aaj ka IPL match fixed tha. Bookies ne players ko 2 crore diye. Insider source se mila — ye channel band hone wala hai. ABHI share karo.",
        "language": "hinglish",
        "topic": "ipl",
        "is_misinfo": True,
        "spread_rate": 0.95,
        "trust_multiplier": 4.5
    },
    {
        "id": "t4_002",
        "text": "IPL fixing proof: ye player ka WhatsApp chat leak hua. Bookie ke saath baat kar raha tha. Cricket boards ko pata hai par chup hain.",
        "language": "hindi",
        "topic": "ipl",
        "is_misinfo": True,
        "spread_rate": 0.9,
        "trust_multiplier": 4.0
    },
    {
        "id": "t4_003",
        "text": "IPL போட்டி ஃபிக்ஸிங் நடந்தது — ஆதாரம் இங்கே உள்ளது. BCCI இதை மூடிமறைக்கிறது. உடனே பகிருங்கள்.",
        "language": "tamil",
        "topic": "ipl",
        "is_misinfo": True,
        "spread_rate": 0.88,
        "trust_multiplier": 3.8
    },
]

TASK5_MISINFO = [
    {
        "id": "t5_001",
        "text": "URGENT: [Community] ke log [Area] mein hamla kar rahe hain. Apne ghar ke mardon ko bahar mat jaane do. Hathiyar lekar taiyaar raho. ABHI share karo.",
        "language": "hinglish",
        "topic": "violence",
        "is_misinfo": True,
        "spread_rate": 0.98,
        "trust_multiplier": 5.0
    },
    {
        "id": "t5_002",
        "text": "அவசரம்: [சமூகம்] மக்கள் தாக்குதல் நடத்துகிறார்கள். உங்கள் குடும்பத்தினரை பாதுகாத்துக்கொள்ளுங்கள். இப்போதே பகிருங்கள்.",
        "language": "tamil",
        "topic": "violence",
        "is_misinfo": True,
        "spread_rate": 0.95,
        "trust_multiplier": 5.0
    },
]

ALL_MISINFO = (
    TASK1_MISINFO +
    TASK2_MISINFO +
    TASK3_MISINFO +
    TASK4_MISINFO +
    TASK5_MISINFO
)

MISINFO_BY_TASK = {
    "task1": TASK1_MISINFO,
    "task2": TASK2_MISINFO,
    "task3": TASK3_MISINFO,
    "task4": TASK4_MISINFO,
    "task5": TASK5_MISINFO,
}

GROUP_TYPES = {
    "family": {
        "name_template": "Family Group 👨‍👩‍👧‍👦",
        "trust_multiplier": 3.0,
        "spread_multiplier": 2.5,
        "skepticism": 0.1
    },
    "locality": {
        "name_template": "RWA / Colony Group",
        "trust_multiplier": 2.0,
        "spread_multiplier": 1.8,
        "skepticism": 0.3
    },
    "college": {
        "name_template": "College Batch Group",
        "trust_multiplier": 1.2,
        "spread_multiplier": 1.0,
        "skepticism": 0.7
    },
    "religious": {
        "name_template": "Religious Group",
        "trust_multiplier": 4.0,
        "spread_multiplier": 3.5,
        "skepticism": 0.05
    },
    "news_channel": {
        "name_template": "News Channel 📢",
        "trust_multiplier": 1.5,
        "spread_multiplier": 5.0,
        "skepticism": 0.2
    },
}


def get_misinfo_for_task(task_id: str) -> List[Dict]:
    return MISINFO_BY_TASK.get(task_id, [])


def get_random_misinfo(task_id: str) -> Dict:
    messages = get_misinfo_for_task(task_id)
    return random.choice(messages)


def get_real_news() -> List[Dict]:
    return REAL_NEWS


def get_random_real_news() -> Dict:
    return random.choice(REAL_NEWS)


def get_group_config(group_type: str) -> Dict:
    return GROUP_TYPES.get(group_type, GROUP_TYPES["locality"])