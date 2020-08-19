require "application_system_test_case"

class UserPersonalsTest < ApplicationSystemTestCase
  setup do
    @user_personal = user_personals(:one)
  end

  test "visiting the index" do
    visit user_personals_url
    assert_selector "h1", text: "User Personals"
  end

  test "creating a User personal" do
    visit user_personals_url
    click_on "New User Personal"

    fill_in "P five", with: @user_personal.p_five
    fill_in "P four", with: @user_personal.p_four
    fill_in "P one", with: @user_personal.p_one
    fill_in "P six", with: @user_personal.p_six
    fill_in "P three", with: @user_personal.p_three
    fill_in "P two", with: @user_personal.p_two
    fill_in "User", with: @user_personal.user
    click_on "Create User personal"

    assert_text "User personal was successfully created"
    click_on "Back"
  end

  test "updating a User personal" do
    visit user_personals_url
    click_on "Edit", match: :first

    fill_in "P five", with: @user_personal.p_five
    fill_in "P four", with: @user_personal.p_four
    fill_in "P one", with: @user_personal.p_one
    fill_in "P six", with: @user_personal.p_six
    fill_in "P three", with: @user_personal.p_three
    fill_in "P two", with: @user_personal.p_two
    fill_in "User", with: @user_personal.user
    click_on "Update User personal"

    assert_text "User personal was successfully updated"
    click_on "Back"
  end

  test "destroying a User personal" do
    visit user_personals_url
    page.accept_confirm do
      click_on "Destroy", match: :first
    end

    assert_text "User personal was successfully destroyed"
  end
end
