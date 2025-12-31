from app import create_app

app = create_app()

if __name__ == "__main__":
    # Debug modu .env dosyasından kontrol edilir
    port = int(app.config.get("PORT", 5001))
    debug = app.config.get("DEBUG") is True
    app.run(host="0.0.0.0", port=port, debug=debug)
