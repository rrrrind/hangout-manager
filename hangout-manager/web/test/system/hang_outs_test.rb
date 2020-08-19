require "application_system_test_case"

class HangOutsTest < ApplicationSystemTestCase
  setup do
    @hang_out = hang_outs(:one)
  end

  test "visiting the index" do
    visit hang_outs_url
    assert_selector "h1", text: "Hang Outs"
  end

  test "creating a Hang out" do
    visit hang_outs_url
    click_on "New Hang Out"

    fill_in "Date", with: @hang_out.date
    fill_in "End time", with: @hang_out.end_time
    fill_in "Name", with: @hang_out.name
    fill_in "Start time", with: @hang_out.start_time
    fill_in "User", with: @hang_out.user_id
    click_on "Create Hang out"

    assert_text "Hang out was successfully created"
    click_on "Back"
  end

  test "updating a Hang out" do
    visit hang_outs_url
    click_on "Edit", match: :first

    fill_in "Date", with: @hang_out.date
    fill_in "End time", with: @hang_out.end_time
    fill_in "Name", with: @hang_out.name
    fill_in "Start time", with: @hang_out.start_time
    fill_in "User", with: @hang_out.user_id
    click_on "Update Hang out"

    assert_text "Hang out was successfully updated"
    click_on "Back"
  end

  test "destroying a Hang out" do
    visit hang_outs_url
    page.accept_confirm do
      click_on "Destroy", match: :first
    end

    assert_text "Hang out was successfully destroyed"
  end
end
