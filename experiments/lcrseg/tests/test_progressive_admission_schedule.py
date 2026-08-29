from lcrseg.methods.components.progressive_admission import site_progress


def test_progressive_admission_schedule() -> None:
    assert site_progress(site_step=0, total_site_steps=100) == 0.0
    assert site_progress(site_step=99, total_site_steps=100) == 1.0
    assert abs((0.4 + 0.4 * site_progress(site_step=49, total_site_steps=99)) - 0.6) < 1.0e-12
