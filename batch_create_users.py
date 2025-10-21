import bcrypt
from pathlib import Path

# Input plain text file (username:password per line)
INPUT_PATH = Path(__file__).parent / "new_users.txt"
# Output hashed file
USERS_PATH = Path(__file__).parent / "users.txt"

def hash_password(pw: str) -> str:
    """Hash the password using bcrypt."""
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def main():
    if not INPUT_PATH.exists():
        print(f"⚠️ No {INPUT_PATH} found. Create it with lines like username:password")
        return

    lines = []
    for raw in INPUT_PATH.read_text().splitlines():
        s = raw.strip()
        if not s or ":" not in s:
            continue
        username, pw = s.split(":", 1)
        username, pw = username.strip(), pw.strip()
        if not username or not pw:
            continue

        hashed = hash_password(pw)
        lines.append(f"{username}:{hashed}")
        print(f"✅ Processed user: {username}")

    if lines:
        USERS_PATH.write_text("\n".join(lines) + "\n")
        print(f"\n🔒 Hashed users saved to {USERS_PATH}")
    else:
        print("⚠️ No valid user entries found in new_users.txt")

if __name__ == "__main__":
    main()
