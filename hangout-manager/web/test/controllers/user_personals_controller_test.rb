require 'test_helper'

class UserPersonalsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @user_personal = user_personals(:one)
  end

  test "should get index" do
    get user_personals_url
    assert_response :success
  end

  test "should get new" do
    get new_user_personal_url
    assert_response :success
  end

  test "should create user_personal" do
    assert_difference('UserPersonal.count') do
      post user_personals_url, params: { user_personal: { p_five: @user_personal.p_five, p_four: @user_personal.p_four, p_one: @user_personal.p_one, p_six: @user_personal.p_six, p_three: @user_personal.p_three, p_two: @user_personal.p_two, user: @user_personal.user } }
    end

    assert_redirected_to user_personal_url(UserPersonal.last)
  end

  test "should show user_personal" do
    get user_personal_url(@user_personal)
    assert_response :success
  end

  test "should get edit" do
    get edit_user_personal_url(@user_personal)
    assert_response :success
  end

  test "should update user_personal" do
    patch user_personal_url(@user_personal), params: { user_personal: { p_five: @user_personal.p_five, p_four: @user_personal.p_four, p_one: @user_personal.p_one, p_six: @user_personal.p_six, p_three: @user_personal.p_three, p_two: @user_personal.p_two, user: @user_personal.user } }
    assert_redirected_to user_personal_url(@user_personal)
  end

  test "should destroy user_personal" do
    assert_difference('UserPersonal.count', -1) do
      delete user_personal_url(@user_personal)
    end

    assert_redirected_to user_personals_url
  end
end
