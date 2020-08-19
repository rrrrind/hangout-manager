require "application_system_test_case"

class UserDetailsTest < ApplicationSystemTestCase
  setup do
    @user_detail = user_details(:one)
  end

  test "visiting the index" do
    visit user_details_url
    assert_selector "h1", text: "User Details"
  end

  test "creating a User detail" do
    visit user_details_url
    click_on "New User Detail"

    fill_in "D eight", with: @user_detail.d_eight
    fill_in "D five", with: @user_detail.d_five
    fill_in "D four", with: @user_detail.d_four
    fill_in "D one", with: @user_detail.d_one
    fill_in "D seven", with: @user_detail.d_seven
    fill_in "D six", with: @user_detail.d_six
    fill_in "D three", with: @user_detail.d_three
    fill_in "D two", with: @user_detail.d_two
    fill_in "User", with: @user_detail.user
    click_on "Create User detail"

    assert_text "User detail was successfully created"
    click_on "Back"
  end

  test "updating a User detail" do
    visit user_details_url
    click_on "Edit", match: :first

    fill_in "D eight", with: @user_detail.d_eight
    fill_in "D five", with: @user_detail.d_five
    fill_in "D four", with: @user_detail.d_four
    fill_in "D one", with: @user_detail.d_one
    fill_in "D seven", with: @user_detail.d_seven
    fill_in "D six", with: @user_detail.d_six
    fill_in "D three", with: @user_detail.d_three
    fill_in "D two", with: @user_detail.d_two
    fill_in "User", with: @user_detail.user
    click_on "Update User detail"

    assert_text "User detail was successfully updated"
    click_on "Back"
  end

  test "destroying a User detail" do
    visit user_details_url
    page.accept_confirm do
      click_on "Destroy", match: :first
    end

    assert_text "User detail was successfully destroyed"
  end
end
