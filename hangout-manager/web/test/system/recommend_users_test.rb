require "application_system_test_case"

class RecommendUsersTest < ApplicationSystemTestCase
  setup do
    @recommend_user = recommend_users(:one)
  end

  test "visiting the index" do
    visit recommend_users_url
    assert_selector "h1", text: "Recommend Users"
  end

  test "creating a Recommend user" do
    visit recommend_users_url
    click_on "New Recommend User"

    fill_in "User", with: @recommend_user.user_id
    click_on "Create Recommend user"

    assert_text "Recommend user was successfully created"
    click_on "Back"
  end

  test "updating a Recommend user" do
    visit recommend_users_url
    click_on "Edit", match: :first

    fill_in "User", with: @recommend_user.user_id
    click_on "Update Recommend user"

    assert_text "Recommend user was successfully updated"
    click_on "Back"
  end

  test "destroying a Recommend user" do
    visit recommend_users_url
    page.accept_confirm do
      click_on "Destroy", match: :first
    end

    assert_text "Recommend user was successfully destroyed"
  end
end
