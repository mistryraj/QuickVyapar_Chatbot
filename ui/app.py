import os
import time
import uuid
import requests
import streamlit as st

API_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="QuickVyapar — Shop Smarter with AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme / CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      :root {
        --bg: #0a0c12;
        --card: #161a23;
        --card-hover: #1e2330;
        --border: rgba(255,255,255,0.08);
        --accent: #22d3ee;
        --accent-2: #4ade80;
        --pink: #f472b6;
        --muted: rgba(250,250,250,0.55);
        --text: #e5e7eb;
      }
      .block-container { padding-top: 0.6rem; padding-bottom: 4rem; max-width: 1180px; }
      [data-testid="stHeader"] { background: transparent; }
      footer, #MainMenu { visibility: hidden; }

      /* Top nav */
      .brand-row { display: flex; align-items: center; gap: 10px; height: 100%; padding-top: 4px; }
      .brand-row .logo {
          background: linear-gradient(135deg, var(--accent), var(--pink));
          width: 32px; height: 32px; border-radius: 8px;
          display: flex; align-items: center; justify-content: center;
          font-weight: 800; color: #0a0c12;
      }
      .brand-row .brand-name { font-weight: 700; font-size: 1.1rem; }
      .nav-divider { border-bottom: 1px solid var(--border); margin: 4px 0 18px; }

      /* Hero */
      .hero {
          padding: 60px 30px; border-radius: 20px; text-align: center; margin-bottom: 28px;
          background:
            radial-gradient(circle at 20% 20%, rgba(244,114,182,0.18), transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(34,211,238,0.18), transparent 50%),
            linear-gradient(180deg, #0e1118, #0a0c12);
          border: 1px solid var(--border);
      }
      .hero h1 { font-size: 2.6rem; margin: 0 0 10px; letter-spacing: -0.5px; }
      .hero .accent { background: linear-gradient(135deg, var(--accent), var(--pink));
                       -webkit-background-clip: text; background-clip: text; color: transparent; }
      .hero p { color: var(--muted); font-size: 1.05rem; max-width: 620px; margin: 0 auto 22px; }

      /* Section header */
      .section-h {
          display: flex; justify-content: space-between; align-items: end; margin: 24px 4px 14px;
      }
      .section-h h2 { margin: 0; font-size: 1.4rem; }
      .section-h p { margin: 2px 0 0; color: var(--muted); font-size: 0.88rem; }

      /* Product card */
      .pcard {
          background: var(--card); border: 1px solid var(--border);
          border-radius: 14px; overflow: hidden; transition: 0.2s; height: 100%;
      }
      .pcard:hover { border-color: var(--accent); transform: translateY(-2px); }
      .pcard .img-wrap {
          position: relative; aspect-ratio: 1/1; background: #0d1018; overflow: hidden;
      }
      .pcard .img-wrap img { width: 100%; height: 100%; object-fit: cover; }
      .pcard .badge {
          position: absolute; top: 8px; left: 8px;
          background: var(--pink); color: #0a0c12; font-size: 0.7rem;
          font-weight: 700; padding: 3px 8px; border-radius: 6px;
      }
      .pcard .body { padding: 12px 14px; }
      .pcard .body h4 { margin: 0 0 4px; font-size: 0.95rem; line-height: 1.25;
                         display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                         overflow: hidden; min-height: 2.4em; }
      .pcard .body .seller { color: var(--muted); font-size: 0.78rem; margin-bottom: 6px; }
      .pcard .body .price { font-size: 1.1rem; font-weight: 700; color: var(--accent-2); }
      .pcard .body .unit { color: var(--muted); font-size: 0.8rem; font-weight: 400; }

      /* Chat */
      [data-testid="stChatMessage"] { padding: 0.55rem 0.75rem; border-radius: 14px; }
      .product-chip {
          display: inline-flex; align-items: center; gap: 0.6rem;
          background: var(--card); border: 1px solid var(--border);
          border-radius: 12px; padding: 6px 10px; margin: 4px 6px 0 0; font-size: 0.82rem;
      }
      .product-chip img { width: 36px; height: 36px; object-fit: cover; border-radius: 8px; }
      .price-tag { color: var(--accent-2); font-weight: 600; }
      .small-muted { color: var(--muted); font-size: 0.78rem; }

      /* Seller card */
      .seller-card {
          background: var(--card); border: 1px solid var(--border);
          border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;
      }
      .seller-card a { color: var(--accent); text-decoration: none; }
      .seller-card a:hover { text-decoration: underline; }

      /* Typing dots */
      .typing { display: inline-block; }
      .typing span {
          height: 7px; width: 7px; margin: 0 2px; background: var(--muted);
          border-radius: 50%; display: inline-block; animation: bounce 1.2s infinite;
      }
      .typing span:nth-child(2) { animation-delay: 0.15s; }
      .typing span:nth-child(3) { animation-delay: 0.30s; }
      @keyframes bounce {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
          40% { transform: scale(1.0); opacity: 1.0; }
      }

      /* Buttons — full hit area */
      .stButton > button {
          border-radius: 10px;
          padding: 10px 16px !important;
          min-height: 42px;
          width: 100%;
          cursor: pointer;
          line-height: 1.2;
          white-space: nowrap;
          background: var(--card);
          border: 1px solid var(--border);
          color: var(--text);
          transition: 0.15s;
      }
      .stButton > button:hover {
          background: var(--card-hover);
          border-color: var(--accent);
          color: var(--accent);
      }
      .stButton > button:active { transform: scale(0.98); }
      .stButton { width: 100%; }

      /* Inline product card inside chat reply */
      .inline-pcard {
          background: var(--card); border: 1px solid var(--border);
          border-radius: 12px; padding: 8px 10px; margin: 6px 0;
          display: flex; gap: 12px; align-items: center;
      }
      .inline-pcard img {
          width: 80px; height: 80px; object-fit: cover; border-radius: 10px; flex-shrink: 0;
      }
      .inline-pcard .meta { flex: 1; min-width: 0; }
      .inline-pcard .meta b { font-size: 0.95rem; }
      .inline-pcard .meta .sub { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }
      .inline-pcard .meta .price { color: var(--accent-2); font-weight: 700; margin-top: 4px; }

      /* Chat input bar */
      [data-testid="stChatInput"] {
          background: var(--card) !important;
          border-radius: 14px !important;
          border: 1px solid var(--border) !important;
          padding: 4px 8px !important;
      }
      [data-testid="stChatInput"]:focus-within {
          border-color: var(--accent) !important;
          box-shadow: 0 0 0 3px rgba(34,211,238,0.15) !important;
      }
      [data-testid="stChatInput"] textarea {
          font-size: 0.95rem !important;
          background: transparent !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ───────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("view", "landing")
ss.setdefault("session_id", str(uuid.uuid4()))
ss.setdefault("messages", [])
ss.setdefault("chat_ended", False)
ss.setdefault("show_sellers", False)
ss.setdefault("pending_input", None)


# ── API helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_greeting() -> str:
    try:
        return requests.get(f"{API_URL}/greeting", timeout=4).json().get("reply", "Hi!")
    except Exception:
        return "Hi 👋 Welcome to QuickVyapar. How can I help you today?"


@st.cache_data(ttl=300)
def fetch_products():
    try:
        return requests.get(f"{API_URL}/products", timeout=4).json().get("data", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_sellers():
    try:
        return requests.get(f"{API_URL}/sellers", timeout=4).json().get("data", [])
    except Exception:
        return []


def call_chat(text: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"session_id": ss.session_id, "message": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"reply": f"Error talking to backend: {e}", "intent": "OFF_TOPIC"}


def goto(view: str):
    ss.view = view


def queue_chat(text: str):
    ss.pending_input = text
    ss.view = "chat"
    if not ss.messages:
        ss.messages.append({"role": "assistant", "content": fetch_greeting(),
                            "meta": {"intent": "GREETING"}})


def reset_chat():
    ss.session_id = str(uuid.uuid4())
    ss.messages = []
    ss.chat_ended = False
    fetch_greeting.clear()


# ── Top nav (single row, flat columns for reliable click targets) ──────────
def _open_chat():
    ss.view = "chat"
    if not ss.messages:
        ss.messages.append({"role": "assistant", "content": fetch_greeting(),
                            "meta": {"intent": "GREETING"}})

show_new_chat_btn = ss.view == "chat"
col_specs = [2, 1, 1, 1] + ([1] if show_new_chat_btn else [])
nav_cols = st.columns(col_specs)

with nav_cols[0]:
    st.markdown(
        "<div class='brand-row'><div class='logo'>Q</div>"
        "<div class='brand-name'>QuickVyapar</div></div>",
        unsafe_allow_html=True,
    )
with nav_cols[1]:
    if st.button("🏠 Home", use_container_width=True, key="nav_home"):
        goto("landing"); st.rerun()
with nav_cols[2]:
    if st.button("🛍️ Products", use_container_width=True, key="nav_prods"):
        goto("products"); st.rerun()
with nav_cols[3]:
    if st.button("🤖 AI Assistant", use_container_width=True, key="nav_ai"):
        _open_chat(); st.rerun()
if show_new_chat_btn:
    with nav_cols[4]:
        if st.button("🧹 New chat", use_container_width=True, key="nav_new"):
            reset_chat(); st.rerun()

st.markdown("<div class='nav-divider'></div>", unsafe_allow_html=True)


# ── Product card renderer ──────────────────────────────────────────────────
def product_card(p: dict, key_prefix: str):
    img = p.get("image") or "https://via.placeholder.com/300x300/0d1018/22d3ee?text=No+Image"
    title = (p.get("title") or "").strip()
    price = p.get("price")
    seller = (p.get("user_name") or "").strip()
    category = (p.get("categoryName") or "").strip()

    st.markdown(
        f"""
        <div class='pcard'>
          <div class='img-wrap'>
            <span class='badge'>NEW</span>
            <img src='{img}' alt='product'/>
          </div>
          <div class='body'>
            <h4>{title}</h4>
            <div class='seller'>{seller} · {category}</div>
            <div class='price'>₹{price} <span class='unit'>{p.get('priceUnitType','') or ''}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🤖 Ask AI about this", key=f"{key_prefix}_{p.get('post_id')}",
                 use_container_width=True):
        queue_chat(f"Tell me about {title}")
        st.rerun()


# ── Landing view ────────────────────────────────────────────────────────────
def render_landing():
    st.markdown(
        """
        <div class='hero'>
          <h1>Shop Smarter with <span class='accent'>AI</span></h1>
          <p>Talk to our smart seller assistant — get product details, prices, sizes, stock, and even negotiate. Available 24/7, grounded in our real catalog.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h1, h2 = st.columns([1, 1])
    with h1:
        if st.button("🛍️ Browse Products", use_container_width=True, type="primary"):
            goto("products"); st.rerun()
    with h2:
        if st.button("🤖 Try AI Assistant", use_container_width=True):
            ss.view = "chat"
            if not ss.messages:
                ss.messages.append({"role": "assistant", "content": fetch_greeting(),
                                    "meta": {"intent": "GREETING"}})
            st.rerun()

    st.markdown(
        "<div class='section-h'><div><h2>✨ Featured Products</h2>"
        "<p>Top picks from our catalog</p></div></div>",
        unsafe_allow_html=True,
    )

    products = fetch_products()
    featured = products[:4]
    cols = st.columns(4)
    for i, p in enumerate(featured):
        with cols[i]:
            product_card(p, "feat")


# ── Products view ───────────────────────────────────────────────────────────
def render_products():
    st.markdown(
        "<div class='section-h'><div><h2>🛍️ All Products</h2>"
        "<p>Click any product to chat with the AI about it</p></div></div>",
        unsafe_allow_html=True,
    )
    products = fetch_products()
    if not products:
        st.info("No products available.")
        return
    for row_start in range(0, len(products), 3):
        row = products[row_start:row_start + 3]
        cols = st.columns(3)
        for i, p in enumerate(row):
            with cols[i]:
                product_card(p, "all")


# ── Chat view ───────────────────────────────────────────────────────────────
def stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.018)


def render_meta(meta: dict, msg_idx: int | None = None):
    """Render reply metadata: negotiation badge, seller notice, product cards.
    msg_idx makes button keys unique across history rerenders.
    """
    if not meta:
        return
    if meta.get("negotiation"):
        n = meta["negotiation"]
        st.caption(
            f"💬 {n['decision'].upper()} · listed ₹{n['listed_price']} · "
            f"offer {'₹'+str(n['buyer_offer']) if n.get('buyer_offer') else '—'} · "
            f"counter {'₹'+str(n['counter_price']) if n.get('counter_price') else '—'} · "
            f"round {n['round_num']}"
        )
    if meta.get("notify_seller"):
        st.success("✅ Seller notified.")

    prods = meta.get("products") or []
    if not prods or meta.get("intent") in (None, "GREETING", "FAREWELL", "OFF_TOPIC"):
        return

    # Single-product detail reply: the text already renders the full product card,
    # so an extra card with another "Ask about this" button is just a redundant
    # loop (clicking it re-asks the same thing). Skip it.
    reply_text = (meta.get("reply") or "").lower()
    if (meta.get("intent") == "PRODUCT_QUERY" and len(prods) == 1
            and (prods[0].get("title") or "").strip().lower() in reply_text):
        return

    # Button only makes sense when there's a list to pick from. For a single
    # product (negotiation context, follow-up) it's the one already in focus —
    # the button would just re-ask about it, so show the card without it.
    show_button = len(prods) > 1

    for i, p in enumerate(prods[:12]):
        img = p.get("image") or "https://via.placeholder.com/80/0d1018/22d3ee?text=No+Image"
        title = (p.get("title") or "").strip()
        price = p.get("price")
        unit = p.get("priceUnitType", "") or ""
        category = (p.get("categoryName", "") or "")[:24]
        seller = (p.get("user_name", "") or "").strip()[:30]

        with st.container(border=True):
            cols = st.columns([1, 3, 2]) if show_button else st.columns([1, 5])
            with cols[0]:
                st.image(img, width=80)
            with cols[1]:
                st.markdown(f"**{title[:60]}**")
                st.caption(f"{seller} · {category}")
                st.markdown(
                    f"<span class='price-tag'>₹{price}</span> "
                    f"<span class='small-muted'>{unit}</span>",
                    unsafe_allow_html=True,
                )
            if show_button:
                with cols[2]:
                    key = f"chat_ask_{msg_idx}_{i}_{p.get('post_id','')}"
                    if st.button("🤖 Ask about this", key=key,
                                 use_container_width=True,
                                 disabled=ss.chat_ended):
                        queue_chat(f"Tell me more about {title}")
                        st.rerun()


def render_chat():
    if not ss.messages:
        ss.messages.append({"role": "assistant", "content": fetch_greeting(),
                            "meta": {"intent": "GREETING"}})

    # Quick actions
    qa = st.columns(4)
    with qa[0]:
        if st.button("📋 All products", use_container_width=True, disabled=ss.chat_ended):
            queue_chat("Show me all products you have"); st.rerun()
    with qa[1]:
        if st.button("💰 Best deals", use_container_width=True, disabled=ss.chat_ended):
            queue_chat("Which are your most affordable products?"); st.rerun()
    with qa[2]:
        if st.button("📞 Sellers", use_container_width=True):
            ss.show_sellers = not ss.show_sellers; st.rerun()
    with qa[3]:
        if st.button("🧵 Cotton tshirts", use_container_width=True, disabled=ss.chat_ended):
            queue_chat("Show me cotton tshirts"); st.rerun()

    if ss.show_sellers:
        with st.expander("📞 Seller contacts", expanded=True):
            for s in fetch_sellers():
                phone_link = f"<a href='tel:{s['phone']}'>{s['phone']}</a>" if s.get("phone") else "—"
                wa_link = (f"<a href='https://wa.me/91{s['whatsapp']}' target='_blank'>{s['whatsapp']}</a>"
                           if s.get("whatsapp") else "—")
                st.markdown(
                    f"<div class='seller-card'><b>{s['name']}</b> "
                    f"<span class='small-muted'>· {s.get('company','')}</span><br/>"
                    f"📱 {phone_link} &nbsp;·&nbsp; 💬 {wa_link}<br/>"
                    f"<span class='small-muted'>📍 {s.get('location','—')} · "
                    f"{s.get('product_count',0)} listing(s)</span></div>",
                    unsafe_allow_html=True,
                )

    # History
    for i, msg in enumerate(ss.messages):
        avatar = "🤖" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            render_meta(msg.get("meta") or {}, msg_idx=i)

    # Pending input → backend → typed reply
    pending = ss.pending_input
    ss.pending_input = None
    if pending and not ss.chat_ended:
        ss.messages.append({"role": "user", "content": pending})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(pending)
        with st.chat_message("assistant", avatar="🤖"):
            ph = st.empty()
            ph.markdown(
                "<div class='typing'><span></span><span></span><span></span></div>",
                unsafe_allow_html=True,
            )
            data = call_chat(pending)
            ph.empty()
            reply = data.get("reply", "")
            st.write_stream(stream_text(reply))
            render_meta(data, msg_idx=len(ss.messages))
        ss.messages.append({"role": "assistant", "content": reply, "meta": data})
        if data.get("end_chat"):
            ss.chat_ended = True
        st.rerun()

    # Composer
    if ss.chat_ended:
        st.info("✅ Chat ended. Click **🧹 New chat** above to start fresh.")
    else:
        prompt = st.chat_input("Ask about a product, price, or make an offer…")
        if prompt:
            queue_chat(prompt); st.rerun()


# ── Router ──────────────────────────────────────────────────────────────────
view = ss.view
if view == "landing":
    render_landing()
elif view == "products":
    render_products()
else:
    render_chat()
