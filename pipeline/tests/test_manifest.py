from sinemart.manifest import build_manifest, parse_entry, pick_headshots

ROOT = "Foto _ Video Peserta/01- Lolos SELEKSI TAHAP AWAL"


def test_parses_real_drive_names():
    r = parse_entry(f"{ROOT}/98 - HAYATUN ROSANA/98-Foto-Closeup-hayatun-rosana.jpg", 129428)
    assert (r.applicant_id, r.kind, r.size) == (98, "closeup", 129428)
    assert r.stage == "01- Lolos SELEKSI TAHAP AWAL"
    r = parse_entry(f"{ROOT}/26 - CITRA NUR AULIA/26-Foto-Fullbody1-citra-nur-aulia.png")
    assert (r.applicant_id, r.kind) == (26, "fullbody1")


def test_skips_videos_and_decks():
    assert parse_entry(f"{ROOT}/98 - HAYATUN ROSANA/Vidio.com - Video dan Tv Streaming di Indonesia.ts") is None
    assert parse_entry("HASIL SELEKSI AWAL/Seleksi Peserta COI 2021 -diva.pptx") is None


def test_renamed_photo_falls_back_to_folder_id():
    r = parse_entry(f"{ROOT}/1114 - DISGA/IMG_2041.JPG")
    assert (r.applicant_id, r.kind) == (1114, "other")


def test_headshot_prefers_closeup():
    listing = [
        {"Path": f"{ROOT}/775 - Defitri salsabilla/775-Foto-Fullbody1-defitri-salsabilla.jpg", "Size": 1},
        {"Path": f"{ROOT}/775 - Defitri salsabilla/775-Foto-Closeup-defitri-salsabilla.jpg", "Size": 1},
        {"Path": f"{ROOT}/164 - Muhammad Reza/164-Foto-Fullbody1-muhammad-reza.jpeg", "Size": 1},
        {"Path": f"{ROOT}/164 - Muhammad Reza", "IsDir": True},
    ]
    heads = pick_headshots(build_manifest(listing))
    assert heads[775].kind == "closeup"
    assert heads[164].kind == "fullbody1"
